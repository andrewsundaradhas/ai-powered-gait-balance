"""
Clinical Recommendations Engine for Gait and Balance Analysis

This module provides evidence-based clinical recommendations based on gait and balance analysis.
It integrates with the analysis results to provide actionable medical insights.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk level classification."""
    NONE = "No Risk"
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    SEVERE = "Severe Risk"

class SeverityLevel(Enum):
    """Severity level classification."""
    NORMAL = "Normal"
    MILD = "Mild"
    MODERATE = "Moderate"
    SEVERE = "Severe"
    CRITICAL = "Critical"


class ClinicalRecommendationEngine:
    """Clinical decision support system for gait and balance analysis."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize with optional configuration."""
        self.protocols = self._load_protocols(config_path)
        self.conditions = self._load_conditions()
        
    def _load_protocols(self, config_path: Optional[str]) -> Dict:
        """Load clinical protocols from configuration."""
        # Default protocols - can be overridden by config file
        protocols = {
            "gait_analysis": {
                "normal": {
                    "description": "Normal gait pattern",
                    "risk_level": RiskLevel.NONE,
                    "severity": SeverityLevel.NORMAL,
                    "interventions": [
                        "Maintain current activity level",
                        "Continue regular exercise routine"
                    ],
                    "follow_up_days": 365,
                    "referral_needed": False,
                },
                "mild_abnormality": {
                    "description": "Mild gait abnormality",
                    "risk_level": RiskLevel.LOW,
                    "severity": SeverityLevel.MILD,
                    "interventions": [
                        "Balance and strength exercises",
                        "Gait training",
                        "Consider physical therapy evaluation"
                    ],
                    "follow_up_days": 180,
                    "referral_needed": False,
                },
                # Add more protocols as needed
            },
            "balance_assessment": {
                "normal": {
                    "description": "Normal balance",
                    "risk_level": RiskLevel.NONE,
                    "severity": SeverityLevel.NORMAL,
                    "interventions": ["Maintain regular physical activity"],
                    "follow_up_days": 365,
                    "referral_needed": False,
                },
                # Add more balance protocols
            }
        }
        
        # Load from config file if provided
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    custom_protocols = json.load(f)
                    protocols.update(custom_protocols)
            except Exception as e:
                logger.warning(f"Failed to load custom protocols: {e}")
                
        return protocols
    
    def _load_conditions(self) -> Dict:
        """Load known medical conditions and their indicators."""
        return {
            "parkinsons_disease": {
                "name": "Parkinson's Disease",
                "indicators": [
                    "bradykinesia",
                    "rigidity",
                    "tremor",
                    "postural_instability",
                    "shuffling_gait",
                    "reduced_arm_swing"
                ],
                "evaluation": self._evaluate_parkinsons,
                "severity_thresholds": {
                    "mild": 0.3,
                    "moderate": 0.6,
                    "severe": 0.8
                }
            },
            "peripheral_neuropathy": {
                "name": "Peripheral Neuropathy",
                "indicators": [
                    "reduced_proprioception",
                    "decreased_vibration_sense",
                    "distal_muscle_weakness",
                    "steppage_gait"
                ],
                "evaluation": self._evaluate_neuropathy,
                "severity_thresholds": {
                    "mild": 0.3,
                    "moderate": 0.6,
                    "severe": 0.8
                }
            },
            # Add more conditions as needed
        }
    
    def assess_gait(self, gait_metrics: Dict[str, Any]) -> ClinicalAssessment:
        """Assess gait based on analysis metrics."""
        # Calculate overall gait score (simplified example)
        gait_score = self._calculate_gait_score(gait_metrics)
        
        # Determine assessment based on score
        if gait_score >= 0.9:
            protocol = self.protocols["gait_analysis"]["normal"]
        elif gait_score >= 0.7:
            protocol = self.protocols["gait_analysis"]["mild_abnormality"]
        else:
            protocol = {
                "description": "Significant gait abnormality",
                "risk_level": RiskLevel.HIGH,
                "severity": SeverityLevel.MODERATE,
                "interventions": [
                    "Urgent referral to neurologist",
                    "Physical therapy evaluation",
                    "Fall risk assessment",
                    "Home safety evaluation"
                ],
                "follow_up_days": 30,
                "referral_needed": True
            }
        
        # Check for specific conditions
        condition_assessments = self._assess_conditions(gait_metrics)
        
        # Combine findings
        key_findings = [f"Gait analysis: {protocol['description']}"]
        key_findings.extend(condition_assessments.get("findings", []))
        
        # Combine interventions
        interventions = protocol["interventions"].copy()
        interventions.extend(condition_assessments.get("interventions", []))
        
        return ClinicalAssessment(
            risk_level=protocol["risk_level"],
            severity=protocol["severity"],
            confidence=0.85,  # Example confidence score
            key_findings=key_findings,
            recommended_actions=interventions,
            follow_up_days=protocol["follow_up_days"],
            requires_urgent_review=protocol.get("referral_needed", False),
            supporting_evidence={
                "gait_score": gait_score,
                "condition_assessments": condition_assessments.get("details", {})
            }
        )
    
    def _calculate_gait_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate an overall gait score from metrics (0-1 scale)."""
        # This is a simplified example - implement based on your specific metrics
        score = 0.0
        total_weight = 0.0
        
        # Example metrics with weights
        metric_weights = {
            "stride_length_variability": 0.3,
            "step_width_variability": 0.25,
            "gait_speed": 0.2,
            "double_support_time": 0.15,
            "swing_time_variability": 0.1
        }
        
        for metric, weight in metric_weights.items():
            if metric in metrics:
                # Normalize to 0-1 scale (higher is better)
                normalized = self._normalize_metric(metric, metrics[metric])
                score += normalized * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def _normalize_metric(self, metric: str, value: float) -> float:
        """Normalize a metric to 0-1 scale (higher is better)."""
        # Define expected ranges for each metric
        ranges = {
            "stride_length_variability": (0, 0.2),  # CV
            "step_width_variability": (0, 0.25),    # CV
            "gait_speed": (0.3, 1.4),              # m/s
            "double_support_time": (0.1, 0.4),      # proportion of gait cycle
            "swing_time_variability": (0, 0.15)     # CV
        }
        
        if metric not in ranges:
            return 0.5  # Default to midpoint if metric not recognized
        
        min_val, max_val = ranges[metric]
        # Clip value to range and normalize to 0-1
        normalized = (max(min(value, max_val), min_val) - min_val) / (max_val - min_val)
        
        # For metrics where lower is better, invert the scale
        if metric in ["stride_length_variability", "step_width_variability", 
                     "double_support_time", "swing_time_variability"]:
            return 1.0 - normalized
        
        return normalized
    
    def _assess_conditions(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess for specific medical conditions."""
        findings = []
        interventions = []
        condition_details = {}
        
        for condition_id, condition in self.conditions.items():
            # Check if we have data for this condition's indicators
            indicators_present = [
                ind for ind in condition["indicators"] 
                if ind in metrics
            ]
            
            if indicators_present:
                # Get condition-specific evaluation
                eval_result = condition["evaluation"](metrics)
                
                if eval_result["present"]:
                    condition_name = condition["name"]
                    severity = eval_result["severity"]
                    confidence = eval_result["confidence"]
                    
                    findings.append(
                        f"Possible {condition_name} ({severity.value.lower()} severity, "
                        f"confidence: {confidence:.1%})"
                    )
                    
                    # Add condition-specific interventions
                    if "interventions" in eval_result:
                        interventions.extend(eval_result["interventions"])
                    
                    # Store detailed information
                    condition_details[condition_id] = {
                        "name": condition_name,
                        "severity": severity.value,
                        "confidence": confidence,
                        "indicators": eval_result.get("indicators", [])
                    }
        
        return {
            "findings": findings,
            "interventions": interventions,
            "details": condition_details
        }
    
    def _evaluate_parkinsons(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate for Parkinson's disease."""
        # Calculate a score based on relevant metrics
        score = 0.0
        max_score = 0.0
        indicators = []
        
        # Bradykinesia (slowness of movement)
        if "gait_speed" in metrics:
            if metrics["gait_speed"] < 0.8:  # m/s
                brady_score = (0.8 - metrics["gait_speed"]) / 0.8
                score += brady_score * 0.4
                max_score += 0.4
                indicators.append({
                    "indicator": "reduced_gait_speed",
                    "value": metrics["gait_speed"],
                    "contribution": brady_score
                })
        
        # Rigidity and reduced arm swing
        if "arm_swing_asymmetry" in metrics:
            arm_swing_score = min(metrics["arm_swing_asymmetry"] / 0.5, 1.0)
            score += arm_swing_score * 0.3
            max_score += 0.3
            indicators.append({
                "indicator": "reduced_arm_swing",
                "value": metrics["arm_swing_asymmetry"],
                "contribution": arm_swing_score
            })
        
        # Postural instability
        if "postural_sway" in metrics:
            sway_score = min(metrics["postural_sway"] / 25.0, 1.0)  # Normalized
            score += sway_score * 0.3
            max_score += 0.3
            indicators.append({
                "indicator": "increased_postural_sway",
                "value": metrics["postural_sway"],
                "contribution": sway_score
            })
        
        # Normalize score
        normalized_score = score / max_score if max_score > 0 else 0.0
        
        # Determine severity
        if normalized_score >= 0.7:
            severity = SeverityLevel.SEVERE
        elif normalized_score >= 0.4:
            severity = SeverityLevel.MODERATE
        else:
            severity = SeverityLevel.MILD
        
        return {
            "present": normalized_score > 0.3,  # Threshold for considering present
            "severity": severity,
            "confidence": min(normalized_score * 1.2, 1.0),  # Cap at 1.0
            "interventions": [
                "Referral to neurologist",
                "Physical therapy evaluation",
                "Consider medication review"
            ],
            "indicators": indicators
        }
    
    def _evaluate_neuropathy(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate for peripheral neuropathy."""
        # Implementation similar to _evaluate_parkinsons
        # ...
        
        return {
            "present": False,  # Placeholder
            "severity": SeverityLevel.NORMAL,
            "confidence": 0.0,
            "interventions": [],
            "indicators": []
        }
    
    def generate_report(self, assessment: ClinicalAssessment) -> str:
        """Generate a human-readable report from the assessment."""
        report = [
            "# Gait and Balance Assessment Report",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            f"**Overall Risk Level:** {assessment.risk_level.value}",
            f"**Severity:** {assessment.severity.value}",
            f"**Confidence:** {assessment.confidence:.1%}",
            "",
            "## Key Findings"
        ]
        
        # Add key findings
        for finding in assessment.key_findings:
            report.append(f"- {finding}")
        
        # Add recommended actions
        report.extend([
            "",
            "## Recommended Actions"
        ])
        
        for action in assessment.recommended_actions:
            report.append(f"- {action}")
        
        # Add follow-up information
        report.extend([
            "",
            "## Follow-up",
            f"Recommended follow-up in {assessment.follow_up_days} days",
            f"Urgent review required: {'Yes' if assessment.requires_urgent_review else 'No'}"
        ])
        
        return "\n".join(report)

    @staticmethod
    def generate_recommendations(
        gait_type: str,
        balance_status: str,
        disorder: str,
        severity: float,
        age: int,
        comorbidities: List[str] | None = None,
    ) -> Dict:
        """Generate clinical recommendations based on analysis results.
        
        Args:
            gait_type: Type of gait abnormality detected
            balance_status: Balance assessment status
            disorder: Detected disorder (if any)
            severity: Severity score (0-1)
            age: Patient age
            comorbidities: List of patient comorbidities
            
        Returns:
            Dict containing structured recommendations
        """
        recommendations = {
            "gait_analysis": {},
            "balance_analysis": {},
            "disorder_management": {},
            "priority_interventions": [],
            "referrals": [],
            "warnings": [],
            "follow_up": {},
            "assistive_devices": [],
            "lifestyle_modifications": []
        }

        engine = ClinicalRecommendationEngine()
        
        # Add gait analysis recommendations
        if gait_type in engine.protocols["gait_analysis"]:
            info = engine.protocols["gait_analysis"][gait_type]
            recommendations["gait_analysis"] = {
                "type": gait_type,
                "description": info["description"],
                "risk_level": info["risk_level"].value,
                "interventions": info["interventions"]
            }
            recommendations["follow_up"]["gait_recheck_days"] = info["follow_up_days"]
            
            # Add interventions to priority list
            recommendations["priority_interventions"].extend(info["interventions"])
            
            # Add referral if needed
            if info.get("referral_needed", False):
                recommendations["referrals"].append("Neurology consultation")

        # Add balance assessment recommendations
        if balance_status in engine.protocols["balance_assessment"]:
            info = engine.protocols["balance_assessment"][balance_status]
            recommendations["balance_analysis"] = {
                "status": balance_status,
                "risk_level": info["risk_level"].value,
                "interventions": info["interventions"]
            }
            recommendations["follow_up"]["balance_recheck_frequency"] = info["follow_up_days"]
            
            # Add interventions to priority list
            recommendations["priority_interventions"].extend(info["interventions"])
        
        # Add disorder management if applicable
        if disorder and disorder.lower() in [cond["name"].lower() for cond in engine.conditions.values()]:
            condition_id = next(
                (cid for cid, cond in engine.conditions.items() 
                 if cond["name"].lower() == disorder.lower()),
                None
            )
            
            if condition_id:
                condition = engine.conditions[condition_id]
                severity_level = (
                    SeverityLevel.SEVERE if severity >= 0.7 else
                    SeverityLevel.MODERATE if severity >= 0.4 else
                    SeverityLevel.MILD if severity >= 0.2 else
                    SeverityLevel.NORMAL
                )
                
                recommendations["disorder_management"] = {
                    "condition": condition["name"],
                    "severity": severity_level.value,
                    "interventions": condition.get("interventions", []),
                    "monitoring_parameters": condition.get("monitoring", [])
                }
                
                # Add disorder-specific interventions
                recommendations["priority_interventions"].extend(condition.get("interventions", []))
                
                # Add high-priority referral for severe cases
                if severity_level in [SeverityLevel.MODERATE, SeverityLevel.SEVERE]:
                    recommendations["referrals"].append(f"{condition['name']} specialist")
        
        # Add age-specific recommendations
        if age >= 65:
            recommendations["warnings"].append(
                "Patient is elderly - increased fall risk. Consider home safety assessment."
            )
            recommendations["assistive_devices"].append(
                "Consider cane or walker for stability"
            )
        
        # Add comorbidity considerations
        if comorbidities:
            if any(d.lower() == "diabetes" for d in comorbidities):
                recommendations["warnings"].append(
                    "Diabetes increases risk of peripheral neuropathy. Monitor for sensory changes."
                )
            if any(h.lower() == "hypertension" for h in comorbidities):
                recommendations["lifestyle_modifications"].append(
                    "Regular blood pressure monitoring recommended"
                )
        
        # Remove duplicates from priority interventions while preserving order
        seen = set()
        recommendations["priority_interventions"] = [
            x for x in recommendations["priority_interventions"] 
            if not (x in seen or seen.add(x))
        ]
        
        return recommendations
				"type": gait_type,
				"description": info["description"],
				"risk_level": info["risk_level"],
				"interventions": info["interventions"],
			}
			recommendations["follow_up"]["gait_recheck_days"] = info["follow_up_days"]

		if balance_status in engine.BALANCE_PROTOCOLS:
			info = engine.BALANCE_PROTOCOLS[balance_status]
			recommendations["balance_analysis"] = {
				"status": balance_status,
				"fall_risk": info["fall_risk"],
				"interventions": info["interventions"],
			}
			recommendations["follow_up"]["balance_recheck_frequency"] = info["monitoring_frequency"]

		if disorder in engine.DISORDER_MANAGEMENT:
			info = engine.DISORDER_MANAGEMENT[disorder]
			recommendations["disorder_management"] = {
				"disorder": disorder,
				"indicators": info["key_indicators"],
				"interventions": info["interventions"],
				"recommended_exercises": info["exercises"],
				"monitoring_parameters": info["monitoring"],
			}
			recommendations["referrals"].append(
				{"specialty": "Neurology", "urgency": info["urgency"], "reason": f"{disorder} management"}
			)

		if severity > 0.8:
			recommendations["warnings"].append(
				{"level": "CRITICAL", "message": "High severity - urgent intervention", "action": "Inpatient evaluation"}
			)
			recommendations["follow_up"]["urgent_recheck_days"] = 14
		elif severity > 0.6:
			recommendations["warnings"].append(
				{"level": "HIGH", "message": "Moderate-high severity", "action": "Frequent monitoring"}
			)

		if age > 75:
			recommendations["lifestyle_modifications"].append("Fall prevention program for elderly")
			recommendations["assistive_devices"].append("Personal alert system")

		if comorbidities and "Diabetes" in comorbidities:
			recommendations["priority_interventions"].append("Glucose control optimization")

		recommendations["priority_interventions"] = [
			recommendations["gait_analysis"].get("description", ""),
			recommendations["balance_analysis"].get("fall_risk", ""),
			f"{disorder} management" if disorder != "None" else "",
		]
		recommendations["priority_interventions"] = [p for p in recommendations["priority_interventions"] if p]
		return recommendations

	@staticmethod
	def export_recommendations(recommendations: Dict, output_format: str = "json") -> str:
		if output_format == "json":
			return json.dumps(recommendations, indent=2)
		if output_format == "text":
			text = "CLINICAL RECOMMENDATIONS\n" + ("=" * 50) + "\n\n"
			if recommendations.get("warnings"):
				text += "⚠️  WARNINGS:\n"
				for w in recommendations["warnings"]:
					text += f"  [{w['level']}] {w['message']}\n"
				text += "\n"
			return text
		return json.dumps(recommendations, indent=2)


