from typing import Dict, Any, List, Optional
import logging
from lightmem.configs.logging.utils import get_logger

class MemoryController:
    """
    PFC-inspired Controller for memory decision making.
    Implements the logic to decide between RAG storage and LoRA micro-update.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = get_logger("MemoryController")
        self.config = config or {}
        
        # Hyperparameters for Utility Function
        self.alpha = self.config.get("alpha", 1.0) # Frequency weight
        self.beta = self.config.get("beta", 1.0)  # Importance weight
        self.gamma = self.config.get("gamma", 2.0) # Risk weight
        self.delta = self.config.get("delta", 0.5) # Cost weight
        
        self.threshold_high = self.config.get("threshold_high", 5.0)
        self.threshold_low = self.config.get("threshold_low", 2.0)
        self.threshold_risk = self.config.get("threshold_risk", 0.8)

    def calculate_utility(self, memory_entry: Any, stats: Dict[str, Any]) -> float:
        """
        Calculate the utility score U(m) for a memory entry.
        
        U(m) = alpha * Freq(m) + beta * Imp(m) - gamma * Risk(m) - delta * Cost(m)
        """
        freq_score = stats.get("frequency", 0.0)
        imp_score = stats.get("importance", 0.5) # Default importance
        risk_score = stats.get("risk", 0.0)
        cost_score = stats.get("cost", 0.1) # Estimated training cost
        
        utility = (self.alpha * freq_score + 
                   self.beta * imp_score - 
                   self.gamma * risk_score - 
                   self.delta * cost_score)
        
        self.logger.debug(f"Utility calc: {utility:.2f} (Freq={freq_score}, Imp={imp_score}, Risk={risk_score})")
        return utility

    def decide_action(self, memory_entry: Any, stats: Dict[str, Any]) -> str:
        """
        Decide the action for a memory entry based on utility.
        
        Returns:
            str: "LORA_MICRO_UPDATE", "VECTOR_STORE_ONLY", "OFFLINE_QUEUE", or "DISCARD"
        """
        risk_score = stats.get("risk", 0.0)
        
        # 1. High risk check
        if risk_score > self.threshold_risk:
            self.logger.info(f"Risk {risk_score} > {self.threshold_risk}, routing to OFFLINE_QUEUE")
            return "OFFLINE_QUEUE"
            
        # 2. Utility calculation
        utility = self.calculate_utility(memory_entry, stats)
        
        # 3. Decision routing
        if utility > self.threshold_high:
            self.logger.info(f"Utility {utility:.2f} > {self.threshold_high}, routing to LORA_MICRO_UPDATE")
            return "LORA_MICRO_UPDATE"
        elif self.threshold_low <= utility <= self.threshold_high:
            self.logger.info(f"Utility {utility:.2f} in range, routing to VECTOR_STORE_ONLY")
            return "VECTOR_STORE_ONLY"
        else:
            self.logger.info(f"Utility {utility:.2f} too low, routing to DISCARD")
            return "DISCARD"
