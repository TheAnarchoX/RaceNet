"""
Driving style scorer - Evaluate driving smoothness and efficiency.

Provides:
- Input smoothness metrics
- Efficiency scoring
- Consistency evaluation
- Penalties for poor driving
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from collections import deque


@dataclass
class StyleScorerConfig:
    """Style scorer configuration."""
    # Smoothness weights
    throttle_smoothness_weight: float = 0.3
    steering_smoothness_weight: float = 0.3
    brake_smoothness_weight: float = 0.2
    line_consistency_weight: float = 0.2
    
    # Sample window
    sample_window_s: float = 2.0
    sample_rate_hz: float = 100.0
    
    # Thresholds
    max_throttle_rate: float = 5.0   # Max acceptable throttle change rate
    max_steering_rate: float = 3.0    # Max acceptable steering change rate
    max_brake_rate: float = 8.0       # Max acceptable brake change rate


class DrivingStyleScorer:
    """Evaluates driving style for smoothness and efficiency.
    
    Provides metrics that can be used to reward or penalize
    ML agents for driving behavior beyond just lap time.
    
    Metrics:
    - Input smoothness (throttle, steering, brake)
    - Racing line consistency
    - Overall efficiency score
    """
    
    def __init__(self, config: StyleScorerConfig | None = None):
        """Initialize style scorer.
        
        Args:
            config: Scorer configuration
        """
        self.config = config or StyleScorerConfig()
        
        # Input history
        buffer_size = int(self.config.sample_window_s * self.config.sample_rate_hz)
        self._throttle_history: deque = deque(maxlen=buffer_size)
        self._steering_history: deque = deque(maxlen=buffer_size)
        self._brake_history: deque = deque(maxlen=buffer_size)
        self._lateral_offset_history: deque = deque(maxlen=buffer_size)
        
        # Cumulative scores
        self._total_samples: int = 0
        self._smoothness_sum: float = 0.0
        self._efficiency_sum: float = 0.0
        
        # Penalties
        self._total_penalties: float = 0.0
        self._penalty_count: int = 0
    
    def update(
        self,
        throttle: float,
        steering: float,
        brake: float,
        lateral_offset: float = 0.0,
        dt: float = 0.01,
    ) -> Dict[str, float]:
        """Update scorer with current inputs.
        
        Args:
            throttle: Current throttle (0-1)
            steering: Current steering (-1 to 1)
            brake: Current brake (0-1)
            lateral_offset: Distance from racing line
            dt: Time step
            
        Returns:
            Dictionary of current scores
        """
        # Store in history
        self._throttle_history.append(throttle)
        self._steering_history.append(steering)
        self._brake_history.append(brake)
        self._lateral_offset_history.append(lateral_offset)
        
        # Calculate scores if we have enough data
        scores = self._calculate_scores()
        
        self._total_samples += 1
        
        return scores
    
    def _calculate_scores(self) -> Dict[str, float]:
        """Calculate current driving style scores.
        
        Returns:
            Dictionary of scores (0-1, higher is better)
        """
        scores = {}
        
        # Need minimum data
        if len(self._throttle_history) < 10:
            return {
                "throttle_smoothness": 1.0,
                "steering_smoothness": 1.0,
                "brake_smoothness": 1.0,
                "line_consistency": 1.0,
                "overall": 1.0,
            }
        
        # Throttle smoothness
        throttle_arr = np.array(self._throttle_history)
        throttle_diff = np.abs(np.diff(throttle_arr))
        throttle_rate = np.mean(throttle_diff) * self.config.sample_rate_hz
        scores["throttle_smoothness"] = max(0.0, 1.0 - throttle_rate / self.config.max_throttle_rate)
        
        # Steering smoothness
        steering_arr = np.array(self._steering_history)
        steering_diff = np.abs(np.diff(steering_arr))
        steering_rate = np.mean(steering_diff) * self.config.sample_rate_hz
        scores["steering_smoothness"] = max(0.0, 1.0 - steering_rate / self.config.max_steering_rate)
        
        # Brake smoothness
        brake_arr = np.array(self._brake_history)
        brake_diff = np.abs(np.diff(brake_arr))
        brake_rate = np.mean(brake_diff) * self.config.sample_rate_hz
        scores["brake_smoothness"] = max(0.0, 1.0 - brake_rate / self.config.max_brake_rate)
        
        # Line consistency
        lateral_arr = np.array(self._lateral_offset_history)
        lateral_std = np.std(lateral_arr)
        # Lower std = more consistent
        scores["line_consistency"] = max(0.0, 1.0 - lateral_std / 5.0)
        
        # Calculate overall score
        scores["overall"] = (
            scores["throttle_smoothness"] * self.config.throttle_smoothness_weight +
            scores["steering_smoothness"] * self.config.steering_smoothness_weight +
            scores["brake_smoothness"] * self.config.brake_smoothness_weight +
            scores["line_consistency"] * self.config.line_consistency_weight
        )
        
        # Update cumulative
        self._smoothness_sum += scores["overall"]
        
        return scores
    
    def add_penalty(self, penalty: float, reason: str = "") -> None:
        """Add a driving penalty.
        
        Args:
            penalty: Penalty amount (0-1)
            reason: Reason for penalty
        """
        self._total_penalties += penalty
        self._penalty_count += 1
    
    def get_average_score(self) -> float:
        """Get average overall score.
        
        Returns:
            Average score (0-1)
        """
        if self._total_samples == 0:
            return 1.0
        return self._smoothness_sum / self._total_samples
    
    def get_efficiency_score(
        self,
        lap_time: float,
        theoretical_best: float,
    ) -> float:
        """Calculate efficiency score comparing to theoretical best.
        
        Args:
            lap_time: Actual lap time
            theoretical_best: Best possible lap time
            
        Returns:
            Efficiency score (0-1)
        """
        if theoretical_best <= 0 or lap_time <= 0:
            return 0.0
        
        # How close to theoretical best
        ratio = theoretical_best / lap_time
        return min(1.0, ratio)
    
    def get_combined_score(
        self,
        lap_time: float,
        theoretical_best: float,
        time_weight: float = 0.6,
        style_weight: float = 0.3,
        penalty_weight: float = 0.1,
    ) -> float:
        """Get combined score incorporating time, style, and penalties.
        
        Args:
            lap_time: Actual lap time
            theoretical_best: Theoretical best lap
            time_weight: Weight for lap time score
            style_weight: Weight for style score
            penalty_weight: Weight for penalties
            
        Returns:
            Combined score (0-1)
        """
        time_score = self.get_efficiency_score(lap_time, theoretical_best)
        style_score = self.get_average_score()
        
        # Penalty score (higher penalties = lower score)
        penalty_score = max(0.0, 1.0 - self._total_penalties)
        
        return (
            time_score * time_weight +
            style_score * style_weight +
            penalty_score * penalty_weight
        )
    
    def reset(self) -> None:
        """Reset all scores."""
        self._throttle_history.clear()
        self._steering_history.clear()
        self._brake_history.clear()
        self._lateral_offset_history.clear()
        self._total_samples = 0
        self._smoothness_sum = 0.0
        self._efficiency_sum = 0.0
        self._total_penalties = 0.0
        self._penalty_count = 0
    
    def get_state(self) -> dict:
        """Get scorer state.
        
        Returns:
            Dictionary with current state
        """
        current_scores = self._calculate_scores()
        
        return {
            "total_samples": self._total_samples,
            "average_score": self.get_average_score(),
            "current_scores": current_scores,
            "total_penalties": self._total_penalties,
            "penalty_count": self._penalty_count,
        }
