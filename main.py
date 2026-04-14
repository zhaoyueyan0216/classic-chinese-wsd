# Test script: Validate the complete pipeline of ancient Chinese word sense disambiguation system using pseudo-ancient Chinese generation

import json
from collections import defaultdict
from embedding_utils import get_embedding, cosine_similarity
from pseudo_retrieval import get_supporting_sentences_with_pseudo_bm25, get_supporting_sentences_without_pseudo_bm25
from prototype import build_prototype
from schema import build_usage_schema
from decision import decide_sense
from llm import call_llm, cached_call_llm


# Base configuration
BASE_CONFIG = {
    "use_top2_filter": True,
    "use_pseudo_query": True,
    "use_bm25": True,
    "use_rerank": True,
    "use_schema": True,
    "use_similarity": True,
    "use_supporting_sentences": True,
    "final_decision_with_llm": True,
    "use_final_cache": False
}

# Different experiment configurations
CONFIGS = {
    # Full configuration
    "full": BASE_CONFIG.copy(),
    
    # Ablation 1: No Top-2 filtering
    "ablation_no_top2": BASE_CONFIG.copy(),
    
    # Ablation 2: Only Top-2
    "ablation_top2_only": BASE_CONFIG.copy(),
    
    # Ablation 3: No historical evidence
    "ablation_no_history": BASE_CONFIG.copy(),
    
    # Ablation 4: No pseudo-ancient Chinese
    "ablation_no_pseudo": BASE_CONFIG.copy(),
    
    # Ablation 5: No schema
    "ablation_no_schema": BASE_CONFIG.copy(),
    
    # Ablation 6: No similarity
    "ablation_no_similarity": BASE_CONFIG.copy()
}

# Set ablation configurations
CONFIGS["ablation_no_top2"]["use_top2_filter"] = False
CONFIGS["ablation_no_top2"]["use_final_cache"] = False
CONFIGS["ablation_top2_only"]["use_supporting_sentences"] = False
CONFIGS["ablation_top2_only"]["use_schema"] = False
CONFIGS["ablation_top2_only"]["use_similarity"] = False
CONFIGS["ablation_top2_only"]["use_final_cache"] = False
CONFIGS["ablation_no_history"]["use_supporting_sentences"] = False
CONFIGS["ablation_no_history"]["use_schema"] = False
CONFIGS["ablation_no_history"]["use_similarity"] = False
CONFIGS["ablation_no_history"]["use_final_cache"] = False
CONFIGS["ablation_no_pseudo"]["use_pseudo_query"] = False
CONFIGS["ablation_no_pseudo"]["use_final_cache"] = False
CONFIGS["ablation_no_schema"]["use_schema"] = False
CONFIGS["ablation_no_schema"]["use_final_cache"] = False
CONFIGS["ablation_no_similarity"]["use_similarity"] = False
CONFIGS["ablation_no_similarity"]["use_final_cache"] = False


def load_test_data():
    """
    Load test data
    
    Returns:
        tuple: (contexts, senses, wsid_to_context, wsid_to_sense, word_to_senses, preloaded_corpus)
    """
    # Load test contexts
    with open("data/character/character_contexts_merged_genres_sample.json", "r", encoding="utf-8") as f:
        contexts = json.load(f)
    
    # Load test senses
    with open("data/character/character_senses.json", "r", encoding="utf-8") as f:
        senses = json.load(f)["RECORDS"]
    
    # Build mapping from wsid to context
    wsid_to_context = {}
    for ctx in contexts:
        wsid = str(ctx["wsid"])
        wsid_to_context[wsid] = ctx
    
    # Build mapping from wsid to sense
    wsid_to_sense = {}
    for sense in senses:
        wsid = str(sense["wsid"])
        wsid_to_sense[wsid] = sense
    
    # Build mapping from word to senses
    word_to_senses = defaultdict(list)
    for sense in senses:
        word = sense["word"]
        word_to_senses[word].append(sense)
    
    # Preloaded corpus is not needed due to Pyserini BM25 retrieval
    preloaded_corpus = []
    
    return contexts, senses, wsid_to_context, wsid_to_sense, word_to_senses, preloaded_corpus


def llm_generate_topk_senses(text, candidate_senses, k=2):
    """
    Generate Top-k senses using LLM
    
    Args:
        text (str): Current context text
        candidate_senses (list): List of candidate senses
        k (int): Number of senses to return
        
    Returns:
        list: List of Top-k senses
    """
    # Three judgments and take union
    all_wsids = set()
    
    for i in range(3):
        print(f"[LLM Judgment #{i+1} - Model: deepseek-v3.2-exp]")
        # Build prompt
        sense_block = "\n".join(
            f"[{i+1}] wsid={s['wsid']}: {s['newgloss']}"
            for i, s in enumerate(candidate_senses)
        )
        
        prompt = f"""
        You are performing an [Ancient Chinese Word Sense Disambiguation Task].
        
        [Target Sentence]
        {text}
        
        [Candidate Senses]
        {sense_block}
        
        Please complete the following tasks:
        1. Select the [most likely {k} senses] from the above options;
        2. No final judgment required, only need to be "reasonable in the current context";
        3. Provide a brief reason for each selected sense.
        
        [Output Format (strict JSON)]
        {
          "top_senses": [
            {
              "wsid": "...",
              "reason": "..."
            }
          ]
        }
        """
        
        # Extract target word from candidate senses
        target_word = candidate_senses[0]['word'] if candidate_senses else "unknown"
        # Call LLM with cache
        cache_tag = f"topk|word={target_word}|run={i}"
        response = cached_call_llm(prompt, cache_tag=cache_tag, seed=42 + i)
        
        # Remove markdown code block markers (if present)
        if response.startswith('```json'):
            response = response[7:]
        if response.endswith('```'):
            response = response[:-3]
        # Remove possible leading/trailing spaces and newlines
        response = response.strip()
        
        try:
            # Parse JSON
            result = json.loads(response)
            # Extract wsid of Top-k senses
            for item in result.get("top_senses", []):
                all_wsids.add(str(item["wsid"]))
            print(f"[LLM Judgment #{i+1}] Successfully extracted {len(result.get('top_senses', []))} senses")
        except Exception as e:
            print(f"[LLM Judgment #{i+1}] Failed to parse response: {e}")
            continue
    
    print(f"[Three judgments result] Union size: {len(all_wsids)}")
    # Convert to sense list
    union_senses = [s for s in candidate_senses if str(s["wsid"]) in all_wsids]
    return union_senses


def process_batch(batch_id, batch_senses, wsid_to_context, wsid_to_sense, test_word_to_senses, sense_embedding_cache, preloaded_corpus, config=None):
    """
    Process a batch of test cases
    
    Args:
        batch_id (int): Batch ID
        batch_senses (list): List of senses in the batch
        wsid_to_context (dict): Mapping from wsid to context
        wsid_to_sense (dict): Mapping from wsid to sense
        test_word_to_senses (dict): Mapping from test word to senses
        sense_embedding_cache (dict): Sense embedding cache
        preloaded_corpus (list): Preloaded corpus
        config (dict): Configuration options
        
    Returns:
        tuple: (correct, total, test_results)
    """
    # Use default configuration
    if config is None:
        config = BASE_CONFIG
    correct = 0
    total = 0
    test_results = []
    
    print(f"\n\n=== Starting processing batch {batch_id+1}, total {len(batch_senses)} test cases ===")
    
    for i, sense in enumerate(batch_senses):
        wsid = str(sense["wsid"])
        word = sense["word"]
        
        # Find corresponding context
        context = wsid_to_context.get(wsid)
        if not context:
            continue  # No corresponding context, skip
        
        total += 1
        print(f"\n\n--- Test sense {i+1}/{len(batch_senses)} (batch {batch_id+1}) ---")
        
        text = context["txt"]
        gold_wsid = wsid
        
        print(f"Target word: {word}")
        print(f"Context: {text}")
        print(f"Gold wsid: {gold_wsid}")
        print(f"Gold sense: {sense['newgloss']}")
        
        # Get all candidate senses for this word (from test word set)
        candidate_senses = test_word_to_senses.get(word, [])
        if not candidate_senses:
            print(f"Warning: Cannot find candidate senses for word={word}, skipping this test")
            continue
        
        print(f"Number of candidate senses: {len(candidate_senses)}")
        for s in candidate_senses:
            print(f"  - wsid={s['wsid']} | gloss={s['newgloss']}")
        
        # Step 1: LLM generates Top-2 senses (three judgments, take union)
        print("\n>>> Step 1: LLM generates Top-2 senses (three judgments, take union)")
        
        if config["use_top2_filter"]:
            try:
                top2_senses = llm_generate_topk_senses(text, candidate_senses, k=2)
                print(f"Three judgments union result:")
                for s in top2_senses:
                    print(f"  - wsid={s['wsid']} | gloss={s['newgloss']}")
            except Exception as e:
                print(f"Error generating Top-2 senses: {e}")
                continue
            
            if len(top2_senses) < 1:
                print("Warning: Insufficient Top-2 senses generated, using alternative strategy")
                # Alternative strategy: use first 2 candidate senses
                if len(candidate_senses) >= 1:
                    top2_senses = candidate_senses[:2]
                    print("Using first 2 candidate senses, continuing test")
                    for s in top2_senses:
                        print(f"  - wsid={s['wsid']} | gloss={s['newgloss']}")
                else:
                    print("Insufficient candidate senses, skipping this test")
                    continue
        else:
            # Not using Top-2 filtering, use all candidate senses
            top2_senses = candidate_senses
            print(f"Not using Top-2 filtering, using all {len(top2_senses)} candidate senses")
        
        # Step 2: Build historical evidence (using pseudo-ancient Chinese generation + Pyserini BM25 retrieval + embedding reranking)
        print("\n>>> Step 2: Build historical evidence")
        context_emb = get_embedding(text)
        sense_evidence_list = []
        
        for sense in top2_senses:
            sense_wsid = str(sense["wsid"])
            sense_gloss = sense["newgloss"]
            print(f"\n--- Processing sense wsid={sense_wsid} ---")
            
            # Initialize evidence
            prototype_vector = None
            schema_summary = ""
            supporting_sentences = []
            similarity = 0.0
            
            if config["use_supporting_sentences"]:
                # Use pseudo-ancient Chinese generation + Pyserini BM25 retrieval + embedding reranking to retrieve supporting sentences
                try:
                    # Define index directory
                    index_dir = r"/mimer/NOBACKUP/groups/cik_data/yueyan/testfrozen_pipeline/bm25_index_zh"
                    
                    if config["use_pseudo_query"]:
                        # Use pseudo-ancient Chinese generation
                        support_sents = get_supporting_sentences_with_pseudo_bm25(
                            sense_gloss=sense_gloss,
                            target_word=word,
                            index_dir=index_dir,
                            wsid=sense_wsid,
                            bm25_top_k=200,
                            final_top_k=10,  # Increase number to get more supporting sentences
                            threshold=0.5
                        )
                    else:
                        # Not using pseudo-ancient Chinese generation, directly use original sense gloss
                        support_sents = get_supporting_sentences_without_pseudo_bm25(
                            sense_gloss=sense_gloss,
                            target_word=word,
                            index_dir=index_dir,
                            wsid=sense_wsid,
                            bm25_top_k=200,
                            final_top_k=10,  # Increase number to get more supporting sentences
                            threshold=0.5
                        )
                    print(f"Retrieved {len(support_sents)} supporting sentences")
                    supporting_sentences = support_sents
                    
                    # Build prototype vector
                    sent_texts = [s["sentence"] for s in support_sents]
                    sent_embs = [get_embedding(s) for s in sent_texts]
                    
                    if sent_embs:
                        prototype_vector = build_prototype(sent_embs)
                        
                        # Calculate similarity with context
                        if config["use_similarity"]:
                            sim = cosine_similarity(context_emb, prototype_vector)
                            similarity = sim
                            print(f"Prototype similarity with context: {sim:.4f}")
                    
                    # Generate usage schema
                    if config["use_schema"]:
                        schema = build_usage_schema(sent_texts, target_word=word)
                        schema_summary = schema
                        print(f"Generated Usage Schema: {schema}")
                    
                except Exception as e:
                    print(f"Error processing sense: {e}")
            else:
                print("Not using supporting sentences")
            
            # Build prototype dictionary with historical evidence text
            sense_evidence_list.append({
                "wsid": sense_wsid,
                "gloss": sense_gloss,
                "prototype_vector": prototype_vector,
                "schema_summary": schema_summary,
                "supporting_sentences": supporting_sentences,
                "similarity": similarity
            })
        
        # Step 3: Final decision
        print("\n>>> Step 3: Final decision")
        
        if config["final_decision_with_llm"]:
            try:
                # Build evidence block
                evidence_block = ""
                for i, evidence in enumerate(sense_evidence_list):
                    evidence_block += f"\n[Candidate Sense {i+1}]"
                    evidence_block += f"\nwsid: {evidence['wsid']}"
                    evidence_block += f"\nSense definition: {evidence['gloss']}"
                    if config["use_similarity"]:
                        evidence_block += f"\nSimilarity with context: {evidence['similarity']:.4f}"
                    if config["use_schema"]:
                        evidence_block += f"\nUsage Schema: {evidence['schema_summary']}"
                    if config["use_supporting_sentences"] and evidence['supporting_sentences']:
                        evidence_block += f"\nSupporting sentences:"
                        for j, sent in enumerate(evidence['supporting_sentences'][:2]):  # Only show first 2 supporting sentences
                            evidence_block += f"\n  {j+1}. {sent['sentence']}"
                    else:
                        evidence_block += "\nSupporting sentences: None"
                    evidence_block += "\n"
                
                # Build final decision prompt
                final_prompt = f"""
                You are an ancient Chinese word sense judgment expert.
                
                Target sentence:
                {text}
                
                Candidate senses and historical evidence:
                {evidence_block}
                
                Please:
                1. Judge whether each candidate sense is reasonable;
                2. Exclude obviously unreasonable senses;
                3. Select the most appropriate sense from the candidates;
                4. Provide a natural language reason.
                
                Output strict JSON:
                {{
                    "prediction": "wsid",
                    "excluded": ["wsid", ...],
                    "reason": "..."
                }}
                """
                
                # Use cached LLM call for final decision
                if config.get("use_final_cache", True):
                    # Improve cache tag, include configuration information to avoid cache collision between different ablation experiments
                    config_signature = "|".join([f"{k}={v}" for k, v in sorted(config.items()) if k != "use_final_cache"])
                    cache_tag = f"final|config={config_signature}|word={word}|context={text[:50]}"
                    response = cached_call_llm(final_prompt, cache_tag=cache_tag)
                else:
                    # Not using cache, directly call LLM
                    print("[Final Decision] Not using cache, directly calling LLM")
                    response = call_llm(final_prompt)
                
                # Remove markdown code block markers (if present)
                if response.startswith('```json'):
                    response = response[7:]
                if response.endswith('```'):
                    response = response[:-3]
                # Remove possible leading/trailing spaces and newlines
                response = response.strip()
                
                # Parse JSON
                final_decision = json.loads(response)
                
                print(f"Final prediction: wsid={final_decision['prediction']}")
                print(f"Excluded senses: {final_decision.get('excluded', [])}")
                print(f"Decision reason: {final_decision['reason']}")
                
                # Verify prediction result
                is_correct = str(final_decision['prediction']) == gold_wsid
                if is_correct:
                    print("✅ Prediction correct")
                    correct += 1
                else:
                    print(f"❌ Prediction incorrect, correct answer is: {gold_wsid}")
                
                # Collect test results
                test_results.append({
                    'Target word': word,
                    'Context': text,
                    'Gold wsid': gold_wsid,
                    'Gold sense': sense['newgloss'],
                    'Predicted wsid': final_decision['prediction'],
                    'Prediction result': 'Correct' if is_correct else 'Incorrect',
                    'Excluded senses': final_decision.get('excluded', []),
                    'Decision reason': final_decision['reason'],
                    'Number of candidate senses': len(candidate_senses),
                    'Number of LLM-generated Top-2 senses': len(top2_senses),
                    'LLM-generated Top-1 sense': top2_senses[0]['wsid'] if len(top2_senses) > 0 else ''
                })
                
            except Exception as e:
                print(f"Error in final decision: {e}")
                # Collect error result
                test_results.append({
                    'Target word': word,
                    'Context': text,
                    'Gold wsid': gold_wsid,
                    'Gold sense': sense['newgloss'],
                    'Predicted wsid': 'Error',
                    'Prediction result': 'Incorrect',
                    'Excluded senses': [],
                    'Decision reason': f'Error: {str(e)}',
                    'Number of candidate senses': len(candidate_senses),
                    'Number of LLM-generated Top-2 senses': len(top2_senses),
                    'LLM-generated Top-1 sense': top2_senses[0]['wsid'] if len(top2_senses) > 0 else ''
                })
                continue
        else:
            # Not using LLM for final decision, use most similar sense
            print("Not using LLM for final decision, using most similar sense")
            if sense_evidence_list:
                # Sort by similarity
                sorted_evidence = sorted(sense_evidence_list, key=lambda x: x['similarity'], reverse=True)
                best_evidence = sorted_evidence[0]
                predicted_wsid = best_evidence['wsid']
                
                print(f"Final prediction: wsid={predicted_wsid} (highest similarity)")
                
                # Verify prediction result
                is_correct = str(predicted_wsid) == gold_wsid
                if is_correct:
                    print("✅ Prediction correct")
                    correct += 1
                else:
                    print(f"❌ Prediction incorrect, correct answer is: {gold_wsid}")
                
                # Collect test results
                test_results.append({
                    'Target word': word,
                    'Context': text,
                    'Gold wsid': gold_wsid,
                    'Gold sense': sense['newgloss'],
                    'Predicted wsid': predicted_wsid,
                    'Prediction result': 'Correct' if is_correct else 'Incorrect',
                    'Excluded senses': [],
                    'Decision reason': f'Using most similar sense, similarity: {best_evidence["similarity"]:.4f}',
                    'Number of candidate senses': len(candidate_senses),
                    'Number of LLM-generated Top-2 senses': len(top2_senses),
                    'LLM-generated Top-1 sense': top2_senses[0]['wsid'] if len(top2_senses) > 0 else ''
                })
            else:
                print("No candidate senses, skipping this test")
                continue
    
    print(f"\n=== Batch {batch_id+1} processing completed ===")
    print(f"Batch {batch_id+1} result: {correct}/{total} = {correct/total:.3f}")
    
    return correct, total, test_results

def run_test(config_name="full"):
    """
    Run complete test process
    
    Args:
        config_name (str): Configuration name, optional values: full, ablation_no_top2, ablation_top2_only, ablation_no_history, ablation_no_pseudo, ablation_no_schema, ablation_no_similarity
    """
    # Get configuration
    config = CONFIGS.get(config_name, BASE_CONFIG)
    
    print("=" * 90)
    print(f"Ancient Chinese Word Sense Disambiguation System Test - Using Pseudo-Ancient Chinese Generation")
    print(f"Configuration: {config_name}")
    print("=" * 90)
    
    # Print configuration details
    print("Configuration details:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 90)
    
    # Load test data
    contexts, all_senses, wsid_to_context, wsid_to_sense, all_word_to_senses, preloaded_corpus = load_test_data()
    
    # Only use senses corresponding to 1000 contexts for testing
    sampled_senses = []
    context_wsids = set()
    for ctx in contexts:
        wsid = str(ctx["wsid"])
        context_wsids.add(wsid)
        sense = wsid_to_sense.get(wsid)
        if sense:
            sampled_senses.append(sense)
    
    # Process all test cases
    print(f"Processing all test cases: total {len(sampled_senses)}")
    
    # Extract target word set for testing (from all contexts)
    test_words = set()
    for sense in sampled_senses:
        test_words.add(sense["word"])
    
    # Sort test words to ensure consistent order
    sorted_test_words = sorted(test_words)
    
    # Build test word_to_senses mapping (only for test words)
    test_word_to_senses = {}
    for word in sorted_test_words:
        test_word_to_senses[word] = all_word_to_senses[word]
    
    # Output test information
    print(f"Loaded {len(contexts)} test contexts")
    print(f"Using {len(sampled_senses)} senses for testing")
    print(f"Actual number of target words to test: {len(sorted_test_words)} (sorted)")
    print(f"Sorted test words: {sorted_test_words}")
    
    # Initialize empty sense_embedding_cache to maintain function signature
    sense_embedding_cache = {}
    
    # Batch processing: split 1000 test cases into 10 batches, 100 cases per batch
    batch_size = 100
    batches = []
    for i in range(0, len(sampled_senses), batch_size):
        batches.append(sampled_senses[i:i+batch_size])
    
    print(f"\n>>> Starting batch processing, total {len(batches)} batches, {batch_size} test cases per batch")
    
    # Process batches serially
    total_correct = 0
    total_total = 0
    all_test_results = []
    
    for batch_id, batch_senses in enumerate(batches):
        try:
            correct, total, test_results = process_batch(
                batch_id,
                batch_senses,
                wsid_to_context,
                wsid_to_sense,
                test_word_to_senses,
                sense_embedding_cache,
                preloaded_corpus,
                config=config
            )
            total_correct += correct
            total_total += total
            all_test_results.extend(test_results)
            print(f"\n=== Batch {batch_id+1} completed, cumulative result: {total_correct}/{total_total} = {total_correct/total_total:.3f} ===")
        except Exception as e:
            print(f"Batch {batch_id+1} processing failed: {e}")
    
    # Export test results to Excel
    print("\n>>> Exporting test results to Excel")
    try:
        import pandas as pd
        from datetime import datetime
        
        # Create DataFrame
        df = pd.DataFrame(all_test_results)
        
        # Add statistics row
        if total_total > 0:
            accuracy = total_correct / total_total
        else:
            accuracy = 0.0
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create statistics dictionary
        stats_row = {
            'Target word': 'Statistics',
            'Context': '',
            'Gold wsid': '',
            'Gold sense': '',
            'Predicted wsid': '',
            'Prediction result': '',
            'Excluded senses': '',
            'Decision reason': '',
            'Number of candidate senses': total_total,
            'Number of LLM-generated Top-2 senses': total_correct,
            'LLM-generated Top-1 sense': f'{accuracy:.3f}'
        }
        
        # Add statistics row to DataFrame (using pd.concat instead of deprecated append method)
        stats_df = pd.DataFrame([stats_row])
        df = pd.concat([df, stats_df], ignore_index=True)
        
        # Export to Excel
        excel_file = f'pseudo_test_results_{config_name}_{timestamp}.xlsx'
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"✅ Successfully exported test results to {excel_file}")
        print(f"Exported {len(all_test_results)} test records")
        print(f"Added statistics row: Number of test cases={total_total}, Number of correct predictions={total_correct}, Accuracy={accuracy:.3f}")
        
    except Exception as e:
        print(f"⚠️  Failed to export to Excel: {e}")
        # Try to export to CSV
        try:
            import pandas as pd
            from datetime import datetime
            
            # Ensure timestamp is defined
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Create DataFrame
            df = pd.DataFrame(all_test_results)
            
            # Add statistics row
            if total_total > 0:
                accuracy = total_correct / total_total
            else:
                accuracy = 0.0
            
            # Create statistics dictionary
            stats_row = {
                'Target word': 'Statistics',
                'Context': '',
                'Gold wsid': '',
                'Gold sense': '',
                'Predicted wsid': '',
                'Prediction result': '',
                'Excluded senses': '',
                'Decision reason': '',
                'Number of candidate senses': total_total,
                'Number of LLM-generated Top-2 senses': total_correct,
                'LLM-generated Top-1 sense': f'{accuracy:.3f}'
            }
            
            # Add statistics row to DataFrame
            stats_df = pd.DataFrame([stats_row])
            df = pd.concat([df, stats_df], ignore_index=True)
            
            # Export to CSV
            csv_file = f'pseudo_test_results_{config_name}_{timestamp}.csv'
            df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"✅ Successfully exported test results to {csv_file}")
        except Exception as e:
            print(f"⚠️  Failed to export to CSV as well: {e}")
    
    # Output test results
    print(f"\n\n{'=' * 90}")
    print(f"Test completed: {total_correct}/{total_total} = {total_correct/total_total:.3f}")
    print("=" * 90)


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        if config_name not in CONFIGS:
            print(f"Error: Configuration name '{config_name}' does not exist")
            print(f"Available configuration names: {list(CONFIGS.keys())}")
            sys.exit(1)
        run_test(config_name)
    else:
        # Default to running full configuration
        run_test()