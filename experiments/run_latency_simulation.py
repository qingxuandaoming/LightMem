import time
import random
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from lightmem.memory.controller import MemoryController
from lightmem.memory.lora_manager import LoRAManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Experiment")

def run_experiment():
    """
    Simulates the pipeline to measure latency and verify logic flow.
    """
    logger.info("Starting Experiment: LoRA-Augmented Memory Latency Test")
    
    # Initialize components
    controller = MemoryController(config={
        "alpha": 1.0, "beta": 1.0, "gamma": 2.0, "delta": 0.5,
        "threshold_high": 3.0
    })
    lora_manager = LoRAManager(base_model_path="gpt2", adapter_store_path="./adapters")
    
    # Test Data: 50 Counterfactual Facts
    test_cases = [
        {"fact": f"Fact {i}: LightMem won the best paper award in 2025.", "topic": "LightMem Award", "id": i}
        for i in range(50)
    ]
    
    results = {
        "baseline_latency": [],
        "ours_latency": [],
        "decisions": {"LORA": 0, "VECTOR": 0, "DISCARD": 0, "OFFLINE": 0}
    }
    
    print(f"\n{'='*20} Processing 50 Facts {'='*20}")
    
    for case in test_cases:
        # Simulate stats
        stats = {
            "frequency": random.uniform(0, 5), # Random frequency
            "importance": 0.8,                 # High importance
            "risk": random.uniform(0, 0.5),    # Low risk
            "cost": 0.5
        }
        
        start_time = time.time()
        
        # 1. Controller Decision
        action = controller.decide_action(case, stats)
        results["decisions"][action.split("_")[0]] += 1
        
        # 2. Execution
        if action == "LORA_MICRO_UPDATE":
            # Simulate training trigger (async in real system, but we measure decision cost here)
            # In inference time, we only measure 'Routing + Loading'
            
            # Simulate Retrieval & Routing Latency
            # RAG retrieval ~ 20ms
            # Adapter Selection ~ 5ms
            # Adapter Loading ~ 10-30ms (if cached) or 100ms+ (if cold)
            
            # Here we simulate the inference phase latency
            rag_latency = random.uniform(0.015, 0.025) # 15-25ms
            adapter_load_latency = random.uniform(0.010, 0.020) # 10-20ms
            
            total_latency = rag_latency + adapter_load_latency
            results["ours_latency"].append(total_latency)
            
        elif action == "VECTOR_STORE_ONLY":
            # Baseline RAG latency
            rag_latency = random.uniform(0.015, 0.025) # 15-25ms
            results["baseline_latency"].append(rag_latency)
            # For "Ours" in RAG mode, latency is same as baseline
            results["ours_latency"].append(rag_latency)
            
        # Baseline always uses RAG
        results["baseline_latency"].append(random.uniform(0.015, 0.025))

    # Calculate statistics
    avg_baseline = sum(results["baseline_latency"]) / len(results["baseline_latency"]) * 1000 # ms
    avg_ours = sum(results["ours_latency"]) / len(results["ours_latency"]) * 1000 # ms
    
    print(f"\n{'='*20} Results {'='*20}")
    print(f"Total Cases: {len(test_cases)}")
    print(f"Decisions: {results['decisions']}")
    print(f"Average Baseline Latency (RAG): {avg_baseline:.2f} ms")
    print(f"Average Ours Latency (Dynamic Routing): {avg_ours:.2f} ms")
    print(f"Latency Overhead: {avg_ours - avg_baseline:.2f} ms")
    
    print("\n[NOTE] These are simulated latencies based on component profiling.")
    print("To get real numbers, please run with a real GPU and set mock=False in LoRAManager.")

if __name__ == "__main__":
    run_experiment()
