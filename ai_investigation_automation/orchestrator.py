"""
Result Orchestrator
Aggregates and reconciles results from multiple AI services
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ConflictResolution:
    """Represents a resolved conflict between AI analyses"""
    field_name: str
    conflicting_values: List[Any]
    resolved_value: Any
    resolution_method: str
    confidence: float

@dataclass
class OrchestratedResult:
    """Final orchestrated result combining all AI analyses"""
    personality_profile: Dict[str, Any]
    behavioral_predictions: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    timeline_analysis: Dict[str, Any]
    consensus_metrics: Dict[str, Any]
    conflicts_resolved: List[ConflictResolution]
    metadata: Dict[str, Any]

class ResultOrchestrator:
    """Orchestrates and reconciles results from multiple AI services"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weight_config = config.get("service_weights", {
            "openai": 1.0,
            "anthropic": 1.0,
            "google": 1.0
        })
        
    async def orchestrate_results(self, ai_results: List) -> OrchestratedResult:
        """Main orchestration method"""
        logger.info("Starting result orchestration...")
        
        # Group results by analysis type
        grouped_results = self._group_results_by_type(ai_results)
        
        # Orchestrate each analysis type
        personality_profile = await self._orchestrate_personality_analysis(
            grouped_results.get("personality", [])
        )
        
        behavioral_predictions = await self._orchestrate_behavior_analysis(
            grouped_results.get("behavior_prediction", [])
        )
        
        risk_assessment = await self._orchestrate_risk_analysis(
            grouped_results.get("risk_assessment", [])
        )
        
        timeline_analysis = await self._orchestrate_timeline_analysis(
            grouped_results.get("timeline_analysis", [])
        )
        
        # Calculate consensus metrics
        consensus_metrics = self._calculate_consensus_metrics(grouped_results)
        
        # Collect all conflict resolutions
        all_conflicts = []
        for result_type, results in grouped_results.items():
            conflicts = self._detect_and_resolve_conflicts(results, result_type)
            all_conflicts.extend(conflicts)
        
        return OrchestratedResult(
            personality_profile=personality_profile,
            behavioral_predictions=behavioral_predictions,
            risk_assessment=risk_assessment,
            timeline_analysis=timeline_analysis,
            consensus_metrics=consensus_metrics,
            conflicts_resolved=all_conflicts,
            metadata={
                "orchestration_timestamp": asyncio.get_event_loop().time(),
                "total_ai_results": len(ai_results),
                "successful_analyses": len([r for r in ai_results if r.metadata.get("success", False)]),
                "conflicts_resolved": len(all_conflicts),
                "consensus_score": consensus_metrics.get("overall_consensus", 0.0)
            }
        )
    
    def _group_results_by_type(self, ai_results: List) -> Dict[str, List]:
        """Group AI results by analysis type"""
        grouped = defaultdict(list)
        
        for result in ai_results:
            if hasattr(result, 'analysis_type') and hasattr(result, 'result'):
                grouped[result.analysis_type].append(result)
        
        return dict(grouped)
    
    async def _orchestrate_personality_analysis(self, results: List) -> Dict[str, Any]:
        """Orchestrate personality analysis results"""
        if not results:
            return {}
        
        logger.info("Orchestrating personality analysis...")
        
        # Extract personality traits from all services
        trait_scores = defaultdict(list)
        communication_styles = []
        behavioral_indicators = []
        
        for result in results:
            if not result.metadata.get("success", False):
                continue
                
            service_weight = self.weight_config.get(result.service_name, 1.0)
            result_data = result.result
            
            # Extract Big Five traits (standardized across services)
            if "personality_traits" in result_data:
                traits = result_data["personality_traits"]
                for trait, score in traits.items():
                    if isinstance(score, (int, float)):
                        trait_scores[trait].append((score, service_weight))
            
            # Extract other personality data
            if "personality_profile" in result_data:
                profile = result_data["personality_profile"]
                if "core_traits" in profile:
                    for trait, score in profile["core_traits"].items():
                        if isinstance(score, (int, float)):
                            trait_scores[trait].append((score, service_weight))
            
            # Collect communication styles
            if "communication_style" in result_data:
                communication_styles.append(result_data["communication_style"])
            
            # Collect behavioral indicators
            if "behavioral_indicators" in result_data:
                behavioral_indicators.extend(result_data["behavioral_indicators"])
        
        # Calculate weighted averages for traits
        consolidated_traits = {}
        for trait, scores in trait_scores.items():
            if scores:
                weighted_sum = sum(score * weight for score, weight in scores)
                total_weight = sum(weight for _, weight in scores)
                consolidated_traits[trait] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Consolidate communication styles
        consolidated_communication = self._consolidate_communication_styles(communication_styles)
        
        # Get most common behavioral indicators
        consolidated_indicators = self._get_top_behavioral_indicators(behavioral_indicators)
        
        return {
            "consolidated_traits": consolidated_traits,
            "communication_style": consolidated_communication,
            "behavioral_indicators": consolidated_indicators,
            "trait_confidence": self._calculate_trait_confidence(trait_scores),
            "sources_count": len([r for r in results if r.metadata.get("success", False)])
        }
    
    async def _orchestrate_behavior_analysis(self, results: List) -> Dict[str, Any]:
        """Orchestrate behavioral prediction results"""
        if not results:
            return {}
        
        logger.info("Orchestrating behavioral analysis...")
        
        decision_patterns = []
        interaction_patterns = []
        work_patterns = []
        adaptation_metrics = defaultdict(list)
        
        for result in results:
            if not result.metadata.get("success", False):
                continue
                
            result_data = result.result
            service_weight = self.weight_config.get(result.service_name, 1.0)
            
            # Collect decision patterns
            if "behavioral_predictions" in result_data:
                predictions = result_data["behavioral_predictions"]
                decision_patterns.append(predictions)
            
            if "behavioral_model" in result_data:
                model = result_data["behavioral_model"]
                if "decision_patterns" in model:
                    decision_patterns.append(model["decision_patterns"])
            
            # Collect interaction patterns
            if "interaction_patterns" in result_data:
                interaction_patterns.append(result_data["interaction_patterns"])
            
            if "social_behavior" in result_data.get("behavioral_model", {}):
                interaction_patterns.append(result_data["behavioral_model"]["social_behavior"])
            
            # Collect work patterns
            if "work_patterns" in result_data:
                work_patterns.append(result_data["work_patterns"])
            
            # Collect adaptation capacity metrics
            if "adaptation_capacity" in result_data:
                capacity = result_data["adaptation_capacity"]
                for metric, value in capacity.items():
                    if isinstance(value, (int, float)):
                        adaptation_metrics[metric].append((value, service_weight))
        
        # Consolidate patterns
        consolidated_decision = self._consolidate_patterns(decision_patterns, "decision_making")
        consolidated_interaction = self._consolidate_patterns(interaction_patterns, "interaction")
        consolidated_work = self._consolidate_patterns(work_patterns, "work")
        
        # Calculate weighted adaptation metrics
        consolidated_adaptation = {}
        for metric, values in adaptation_metrics.items():
            if values:
                weighted_sum = sum(value * weight for value, weight in values)
                total_weight = sum(weight for _, weight in values)
                consolidated_adaptation[metric] = weighted_sum / total_weight if total_weight > 0 else 0
        
        return {
            "decision_making": consolidated_decision,
            "interaction_patterns": consolidated_interaction,
            "work_patterns": consolidated_work,
            "adaptation_capacity": consolidated_adaptation,
            "prediction_confidence": self._calculate_prediction_confidence(results),
            "sources_count": len([r for r in results if r.metadata.get("success", False)])
        }
    
    async def _orchestrate_risk_analysis(self, results: List) -> Dict[str, Any]:
        """Orchestrate risk assessment results"""
        if not results:
            return {}
        
        logger.info("Orchestrating risk analysis...")
        
        risk_scores = defaultdict(list)
        risk_categories = defaultdict(list)
        risk_factors = []
        recommendations = []
        
        for result in results:
            if not result.metadata.get("success", False):
                continue
                
            result_data = result.result
            service_weight = self.weight_config.get(result.service_name, 1.0)
            
            # Extract risk scores
            if "risk_assessment" in result_data:
                assessment = result_data["risk_assessment"]
                if "overall_risk_score" in assessment:
                    risk_scores["overall"].append((assessment["overall_risk_score"], service_weight))
                
                # Extract category-specific risks
                for category in ["financial_risk", "reputational_risk", "operational_risk", "compliance_risk"]:
                    if category in assessment:
                        risk_value = self._convert_risk_to_numeric(assessment[category])
                        risk_categories[category].append((risk_value, service_weight))
            
            # Extract comprehensive risk profiles
            if "comprehensive_risk_profile" in result_data:
                profile = result_data["comprehensive_risk_profile"]
                if "risk_categories" in profile:
                    for category, risk_level in profile["risk_categories"].items():
                        risk_value = self._convert_risk_to_numeric(risk_level)
                        risk_categories[category].append((risk_value, service_weight))
            
            # Collect risk factors
            if "risk_factors" in result_data:
                risk_factors.extend(result_data["risk_factors"])
            
            # Collect recommendations
            if "recommendations" in result_data:
                recommendations.extend(result_data["recommendations"])
        
        # Calculate consolidated risk scores
        consolidated_scores = {}
        for risk_type, scores in risk_scores.items():
            if scores:
                weighted_sum = sum(score * weight for score, weight in scores)
                total_weight = sum(weight for _, weight in scores)
                consolidated_scores[risk_type] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate consolidated risk categories
        consolidated_categories = {}
        for category, scores in risk_categories.items():
            if scores:
                weighted_sum = sum(score * weight for score, weight in scores)
                total_weight = sum(weight for _, weight in scores)
                consolidated_categories[category] = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Consolidate risk factors and recommendations
        top_risk_factors = self._consolidate_risk_factors(risk_factors)
        prioritized_recommendations = self._prioritize_recommendations(recommendations)
        
        return {
            "overall_risk_score": consolidated_scores.get("overall", 0),
            "risk_categories": consolidated_categories,
            "top_risk_factors": top_risk_factors,
            "recommendations": prioritized_recommendations,
            "risk_confidence": self._calculate_risk_confidence(results),
            "sources_count": len([r for r in results if r.metadata.get("success", False)])
        }
    
    async def _orchestrate_timeline_analysis(self, results: List) -> Dict[str, Any]:
        """Orchestrate timeline analysis results"""
        if not results:
            return {}
        
        logger.info("Orchestrating timeline analysis...")
        
        career_patterns = []
        milestones = []
        projections = []
        temporal_insights = []
        
        for result in results:
            if not result.metadata.get("success", False):
                continue
                
            result_data = result.result
            
            # Collect career progression patterns
            if "timeline_analysis" in result_data:
                analysis = result_data["timeline_analysis"]
                if "career_progression" in analysis:
                    career_patterns.append(analysis["career_progression"])
                if "major_transitions" in analysis:
                    milestones.extend(analysis["major_transitions"])
            
            # Collect chronological analysis
            if "chronological_analysis" in result_data:
                analysis = result_data["chronological_analysis"]
                if "life_phases" in analysis:
                    milestones.extend(analysis["life_phases"])
            
            # Collect future projections
            if "predictive_timeline" in result_data:
                projections.append(result_data["predictive_timeline"])
            
            if "future_projections" in result_data:
                projections.append(result_data["future_projections"])
            
            # Collect temporal insights
            if "temporal_insights" in result_data:
                temporal_insights.append(result_data["temporal_insights"])
            
            if "temporal_patterns" in result_data:
                temporal_insights.append(result_data["temporal_patterns"])
        
        # Consolidate timeline data
        consolidated_progression = self._consolidate_career_progression(career_patterns)
        consolidated_milestones = self._consolidate_milestones(milestones)
        consolidated_projections = self._consolidate_projections(projections)
        consolidated_insights = self._consolidate_temporal_insights(temporal_insights)
        
        return {
            "career_progression": consolidated_progression,
            "key_milestones": consolidated_milestones,
            "future_projections": consolidated_projections,
            "temporal_patterns": consolidated_insights,
            "timeline_confidence": self._calculate_timeline_confidence(results),
            "sources_count": len([r for r in results if r.metadata.get("success", False)])
        }
    
    def _calculate_consensus_metrics(self, grouped_results: Dict[str, List]) -> Dict[str, Any]:
        """Calculate consensus metrics across all analyses"""
        total_analyses = sum(len(results) for results in grouped_results.values())
        successful_analyses = sum(
            len([r for r in results if r.metadata.get("success", False)])
            for results in grouped_results.values()
        )
        
        # Calculate confidence consensus
        all_confidences = []
        for results in grouped_results.values():
            for result in results:
                if result.metadata.get("success", False):
                    all_confidences.append(result.confidence_score)
        
        avg_confidence = statistics.mean(all_confidences) if all_confidences else 0.0
        confidence_std = statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0.0
        
        # Calculate service agreement
        service_agreement = self._calculate_service_agreement(grouped_results)
        
        return {
            "total_analyses": total_analyses,
            "successful_analyses": successful_analyses,
            "success_rate": successful_analyses / total_analyses if total_analyses > 0 else 0,
            "average_confidence": avg_confidence,
            "confidence_consistency": 1.0 - min(confidence_std / avg_confidence if avg_confidence > 0 else 1.0, 1.0),
            "service_agreement": service_agreement,
            "overall_consensus": (avg_confidence + service_agreement) / 2
        }
    
    def _detect_and_resolve_conflicts(self, results: List, analysis_type: str) -> List[ConflictResolution]:
        """Detect and resolve conflicts between AI service results"""
        conflicts = []
        
        if len(results) < 2:
            return conflicts
        
        # Compare results pairwise for conflicts
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                result1, result2 = results[i], results[j]
                
                if not (result1.metadata.get("success", False) and result2.metadata.get("success", False)):
                    continue
                
                # Find conflicting fields
                field_conflicts = self._find_field_conflicts(
                    result1.result, result2.result, 
                    f"{result1.service_name}_vs_{result2.service_name}"
                )
                conflicts.extend(field_conflicts)
        
        return conflicts
    
    def _find_field_conflicts(self, result1: Dict, result2: Dict, comparison_name: str) -> List[ConflictResolution]:
        """Find conflicts between two result dictionaries"""
        conflicts = []
        
        # Compare numeric values with threshold
        threshold = 0.3  # 30% difference threshold
        
        def compare_nested_dicts(dict1, dict2, path=""):
            for key in set(dict1.keys()) & set(dict2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                    compare_nested_dicts(dict1[key], dict2[key], current_path)
                elif isinstance(dict1[key], (int, float)) and isinstance(dict2[key], (int, float)):
                    if abs(dict1[key] - dict2[key]) > threshold:
                        resolved_value = (dict1[key] + dict2[key]) / 2
                        conflicts.append(ConflictResolution(
                            field_name=current_path,
                            conflicting_values=[dict1[key], dict2[key]],
                            resolved_value=resolved_value,
                            resolution_method="average",
                            confidence=0.7
                        ))
        
        compare_nested_dicts(result1, result2)
        return conflicts
    
    # Helper methods for consolidation
    def _consolidate_communication_styles(self, styles: List[Dict]) -> Dict[str, Any]:
        """Consolidate communication styles from multiple sources"""
        if not styles:
            return {}
        
        # Extract common fields and find most frequent values
        consolidated = {}
        for field in ["formality", "tone", "complexity"]:
            values = [style.get(field) for style in styles if style.get(field)]
            if values:
                consolidated[field] = max(set(values), key=values.count)
        
        return consolidated
    
    def _get_top_behavioral_indicators(self, indicators: List[str], top_n: int = 5) -> List[str]:
        """Get top behavioral indicators by frequency"""
        if not indicators:
            return []
        
        from collections import Counter
        counter = Counter(indicators)
        return [indicator for indicator, _ in counter.most_common(top_n)]
    
    def _consolidate_patterns(self, patterns: List[Dict], pattern_type: str) -> Dict[str, Any]:
        """Consolidate behavioral patterns"""
        if not patterns:
            return {}
        
        consolidated = {}
        all_keys = set()
        for pattern in patterns:
            all_keys.update(pattern.keys())
        
        for key in all_keys:
            values = [pattern.get(key) for pattern in patterns if pattern.get(key)]
            if values:
                if all(isinstance(v, str) for v in values):
                    consolidated[key] = max(set(values), key=values.count)
                elif all(isinstance(v, (int, float)) for v in values):
                    consolidated[key] = statistics.mean(values)
                else:
                    consolidated[key] = values[0]  # Take first non-null value
        
        return consolidated
    
    def _convert_risk_to_numeric(self, risk_level: str) -> float:
        """Convert risk level string to numeric value"""
        risk_mapping = {
            "low": 1.0,
            "medium": 5.0,
            "high": 9.0,
            "very_low": 0.5,
            "very_high": 10.0
        }
        return risk_mapping.get(risk_level.lower(), 5.0)
    
    def _consolidate_risk_factors(self, factors: List[Dict]) -> List[Dict]:
        """Consolidate and prioritize risk factors"""
        if not factors:
            return []
        
        # Group similar factors and prioritize by frequency and impact
        factor_groups = defaultdict(list)
        for factor in factors:
            if isinstance(factor, dict) and "factor" in factor:
                factor_groups[factor["factor"]].append(factor)
        
        consolidated = []
        for factor_name, factor_list in factor_groups.items():
            # Take the factor with highest impact or most recent
            best_factor = max(factor_list, key=lambda x: x.get("impact", "low"))
            consolidated.append(best_factor)
        
        return consolidated[:5]  # Return top 5
    
    def _prioritize_recommendations(self, recommendations: List[str]) -> List[str]:
        """Prioritize recommendations by frequency and importance"""
        if not recommendations:
            return []
        
        from collections import Counter
        counter = Counter(recommendations)
        return [rec for rec, _ in counter.most_common(10)]
    
    def _consolidate_career_progression(self, patterns: List[str]) -> str:
        """Consolidate career progression patterns"""
        if not patterns:
            return "unknown"
        
        from collections import Counter
        counter = Counter(patterns)
        return counter.most_common(1)[0][0] if counter else "steady"
    
    def _consolidate_milestones(self, milestones: List[Dict]) -> List[Dict]:
        """Consolidate timeline milestones"""
        if not milestones:
            return []
        
        # Sort by date and remove duplicates
        unique_milestones = []
        seen_events = set()
        
        for milestone in milestones:
            event_key = milestone.get("event", milestone.get("phase", ""))
            if event_key and event_key not in seen_events:
                unique_milestones.append(milestone)
                seen_events.add(event_key)
        
        return sorted(unique_milestones, key=lambda x: x.get("date", x.get("start_date", "0000")))
    
    def _consolidate_projections(self, projections: List[Dict]) -> Dict[str, Any]:
        """Consolidate future projections"""
        if not projections:
            return {}
        
        consolidated = {}
        for projection in projections:
            for key, value in projection.items():
                if key not in consolidated:
                    consolidated[key] = value
        
        return consolidated
    
    def _consolidate_temporal_insights(self, insights: List[Dict]) -> Dict[str, Any]:
        """Consolidate temporal insights"""
        if not insights:
            return {}
        
        consolidated = {}
        for insight in insights:
            for key, value in insight.items():
                if key not in consolidated:
                    consolidated[key] = value
        
        return consolidated
    
    # Confidence calculation methods
    def _calculate_trait_confidence(self, trait_scores: Dict) -> float:
        """Calculate confidence in personality traits"""
        if not trait_scores:
            return 0.0
        
        confidences = []
        for trait, scores in trait_scores.items():
            if len(scores) > 1:
                values = [score for score, _ in scores]
                std_dev = statistics.stdev(values)
                mean_val = statistics.mean(values)
                confidence = 1.0 - min(std_dev / mean_val if mean_val > 0 else 1.0, 1.0)
                confidences.append(confidence)
        
        return statistics.mean(confidences) if confidences else 0.5
    
    def _calculate_prediction_confidence(self, results: List) -> float:
        """Calculate confidence in behavioral predictions"""
        if not results:
            return 0.0
        
        confidences = [r.confidence_score for r in results if r.metadata.get("success", False)]
        return statistics.mean(confidences) if confidences else 0.0
    
    def _calculate_risk_confidence(self, results: List) -> float:
        """Calculate confidence in risk assessment"""
        if not results:
            return 0.0
        
        confidences = [r.confidence_score for r in results if r.metadata.get("success", False)]
        return statistics.mean(confidences) if confidences else 0.0
    
    def _calculate_timeline_confidence(self, results: List) -> float:
        """Calculate confidence in timeline analysis"""
        if not results:
            return 0.0
        
        confidences = [r.confidence_score for r in results if r.metadata.get("success", False)]
        return statistics.mean(confidences) if confidences else 0.0
    
    def _calculate_service_agreement(self, grouped_results: Dict[str, List]) -> float:
        """Calculate agreement level between different AI services"""
        if not grouped_results:
            return 0.0
        
        agreement_scores = []
        
        for analysis_type, results in grouped_results.items():
            successful_results = [r for r in results if r.metadata.get("success", False)]
            if len(successful_results) > 1:
                # Calculate pairwise agreement
                agreements = []
                for i in range(len(successful_results)):
                    for j in range(i + 1, len(successful_results)):
                        agreement = self._calculate_pairwise_agreement(
                            successful_results[i].result, 
                            successful_results[j].result
                        )
                        agreements.append(agreement)
                
                if agreements:
                    agreement_scores.append(statistics.mean(agreements))
        
        return statistics.mean(agreement_scores) if agreement_scores else 0.5
    
    def _calculate_pairwise_agreement(self, result1: Dict, result2: Dict) -> float:
        """Calculate agreement between two analysis results"""
        # Simplified agreement calculation based on numeric values
        numeric_pairs = []
        
        def extract_numeric_values(d1, d2, path=""):
            for key in set(d1.keys()) & set(d2.keys()):
                if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    extract_numeric_values(d1[key], d2[key], f"{path}.{key}")
                elif isinstance(d1[key], (int, float)) and isinstance(d2[key], (int, float)):
                    numeric_pairs.append((d1[key], d2[key]))
        
        extract_numeric_values(result1, result2)
        
        if not numeric_pairs:
            return 0.5  # Default agreement if no numeric comparisons possible
        
        agreements = []
        for val1, val2 in numeric_pairs:
            if val1 == 0 and val2 == 0:
                agreements.append(1.0)
            else:
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    agreement = 1.0 - min(abs(val1 - val2) / max_val, 1.0)
                    agreements.append(agreement)
        
        return statistics.mean(agreements) if agreements else 0.5
