"""
Scoring system - Combined scoring for ML training.

Provides:
- Unified scoring interface
- Reward calculation for RL
- Performance metrics
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np

from racenet.scoring.lap_timer import LapTimer, LapTimerConfig
from racenet.scoring.style_scorer import DrivingStyleScorer, StyleScorerConfig


@dataclass
class ScoringConfig:
    """Scoring system configuration."""
    # Component configs
    lap_timer: LapTimerConfig | None = None
    style_scorer: StyleScorerConfig | None = None
    
    # Reward weights
    speed_reward_weight: float = 0.3      # Reward for high speed
    progress_reward_weight: float = 0.3   # Reward for track progress
    style_reward_weight: float = 0.2      # Reward for smooth driving
    time_reward_weight: float = 0.2       # Reward for fast lap times
    
    # Penalties
    off_track_penalty: float = 0.5        # Per-instance penalty
    collision_penalty: float = 1.0        # Not used currently (no collisions)
    
    # Normalization
    max_speed_mps: float = 83.33          # ~300 kph for normalization
    target_lap_time_s: float = 90.0       # Expected lap time for normalization


class ScoringSystem:
    """Unified scoring system for ML training.
    
    Combines lap timing, style scoring, and reward calculation
    into a single interface for training ML agents.
    
    Features:
    - Multi-objective scoring
    - Configurable reward weights
    - Real-time reward calculation
    - Episode summary statistics
    """
    
    def __init__(self, config: ScoringConfig | None = None):
        """Initialize scoring system.
        
        Args:
            config: Scoring configuration
        """
        self.config = config or ScoringConfig()
        
        # Components
        self.lap_timer = LapTimer(self.config.lap_timer)
        self.style_scorer = DrivingStyleScorer(self.config.style_scorer)
        
        # Tracking
        self._last_distance: float = 0.0
        self._total_distance: float = 0.0
        self._total_reward: float = 0.0
        self._step_count: int = 0
        
        # Episode stats
        self._episode_rewards: List[float] = []
        self._off_track_count: int = 0
    
    def step(
        self,
        speed: float,
        distance: float,
        throttle: float,
        steering: float,
        brake: float,
        lateral_offset: float,
        is_off_track: bool,
        dt: float,
    ) -> float:
        """Calculate reward for current step.
        
        Args:
            speed: Current speed in m/s
            distance: Distance along track in m
            throttle: Throttle input (0-1)
            steering: Steering input (-1 to 1)
            brake: Brake input (0-1)
            lateral_offset: Lateral offset from center
            is_off_track: Whether car is off track
            dt: Time step
            
        Returns:
            Step reward
        """
        reward = 0.0
        
        # Speed reward (encourage going fast)
        speed_normalized = speed / self.config.max_speed_mps
        speed_reward = speed_normalized * self.config.speed_reward_weight
        reward += speed_reward
        
        # Progress reward (encourage making progress)
        delta_distance = distance - self._last_distance
        if delta_distance < -1000:  # Crossed start/finish
            delta_distance = 0  # Don't penalize for lap completion
        progress_normalized = max(0, delta_distance) / (self.config.max_speed_mps * dt)
        progress_reward = progress_normalized * self.config.progress_reward_weight
        reward += progress_reward
        
        # Style reward
        style_scores = self.style_scorer.update(
            throttle, steering, brake, lateral_offset, dt
        )
        style_reward = style_scores.get("overall", 0.5) * self.config.style_reward_weight
        reward += style_reward
        
        # Off-track penalty
        if is_off_track:
            reward -= self.config.off_track_penalty * dt
            self._off_track_count += 1
            self.lap_timer.record_violation()
            self.style_scorer.add_penalty(0.1, "off_track")
        
        # Update tracking
        self._last_distance = distance
        self._total_distance += max(0, delta_distance)
        self._total_reward += reward
        self._step_count += 1
        self._episode_rewards.append(reward)
        
        return reward
    
    def complete_lap(self, time: float) -> float:
        """Handle lap completion and calculate lap reward.
        
        Args:
            time: Current simulation time
            
        Returns:
            Bonus reward for lap completion
        """
        lap_record = self.lap_timer.complete_lap(time)
        
        # Calculate lap time reward
        time_factor = self.config.target_lap_time_s / lap_record.lap_time_s
        time_factor = np.clip(time_factor, 0.5, 2.0)  # Limit extreme values
        
        lap_reward = time_factor * self.config.time_reward_weight * 10.0  # Scale up
        
        # Bonus for valid lap
        if lap_record.is_valid:
            lap_reward *= 1.2
        
        self._total_reward += lap_reward
        
        return lap_reward
    
    def cross_sector(self, time: float) -> Optional[float]:
        """Handle sector crossing.
        
        Args:
            time: Current simulation time
            
        Returns:
            Sector time if valid
        """
        return self.lap_timer.cross_sector(time)
    
    def start_lap(self, time: float) -> None:
        """Start timing a new lap.
        
        Args:
            time: Current simulation time
        """
        self.lap_timer.start_lap(time)
    
    def get_current_reward(self) -> float:
        """Get total accumulated reward.
        
        Returns:
            Total reward
        """
        return self._total_reward
    
    def get_average_reward(self) -> float:
        """Get average reward per step.
        
        Returns:
            Average step reward
        """
        if self._step_count == 0:
            return 0.0
        return self._total_reward / self._step_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics.
        
        Returns:
            Dictionary of metrics
        """
        return {
            "total_reward": self._total_reward,
            "average_reward": self.get_average_reward(),
            "step_count": self._step_count,
            "total_distance_m": self._total_distance,
            "completed_laps": self.lap_timer.lap_count,
            "valid_laps": self.lap_timer.valid_lap_count,
            "best_lap_time": self.lap_timer.best_lap_time,
            "off_track_count": self._off_track_count,
            "style_score": self.style_scorer.get_average_score(),
            "lap_timer": self.lap_timer.get_state(),
            "style_scorer": self.style_scorer.get_state(),
        }
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get episode summary for logging.
        
        Returns:
            Episode summary dictionary
        """
        rewards = np.array(self._episode_rewards) if self._episode_rewards else np.array([0])
        
        return {
            "total_reward": self._total_reward,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "total_steps": self._step_count,
            "laps_completed": self.lap_timer.lap_count,
            "best_lap_time": self.lap_timer.best_lap_time,
            "average_speed_kph": (self._total_distance / max(1, self._step_count * 0.01)) * 3.6,
        }
    
    def reset(self) -> None:
        """Reset scoring for new episode."""
        self.lap_timer.reset()
        self.style_scorer.reset()
        self._last_distance = 0.0
        self._total_distance = 0.0
        self._total_reward = 0.0
        self._step_count = 0
        self._episode_rewards = []
        self._off_track_count = 0
    
    def get_state(self) -> dict:
        """Get complete scoring state.
        
        Returns:
            Dictionary with all scoring data
        """
        return {
            "step_count": self._step_count,
            "total_reward": self._total_reward,
            "total_distance": self._total_distance,
            "off_track_count": self._off_track_count,
            "lap_timer": self.lap_timer.get_state(),
            "style_scorer": self.style_scorer.get_state(),
        }
