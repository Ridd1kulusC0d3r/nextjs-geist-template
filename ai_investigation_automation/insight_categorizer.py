"""
Insight Categorizer
Categorizes and prioritizes insights from investigation results
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict, Counter
import re
import json

logger = logging.getLogger(__name__)

@dataclass
class CategorizedInsight:
    """Represents a categorized insight"""
    category: str
    subcategory: str
    insight: str
    importance_score: float
    confidence: float
    source_services: List[str]
    supporting_evidence: List[str]
    actionable: bool
    risk_level: str
    metadata: Dict[str, Any]

@dataclass
class InsightCategory:
    """Represents a category of insights"""
    name: str
    description: str
    insights: List[CategorizedInsight]
    priority_score: float
    total_insights: int
    high_confidence_count: int

class InsightCategorizer:
    """Categorizes and prioritizes insights from orchestrated results"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.categories = self._initialize_categories()
        self.importance_weights = self._load_importance_weights()
        
    def _initialize_categories(self) -> Dict[str, Dict[str, Any]]:
        """Initialize insight categories and their definitions"""
        return {
            "personality_traits": {
                "description": "Core personality characteristics and behavioral tendencies",
                "subcategories": [
                    "big_five_traits", "communication_style", "decision_making", 
                    "emotional_intelligence", "leadership_style"
                ],
                "keywords": ["personality", "trait", "behavior", "character", "style"],
                "priority_weight": 0.85
            },
            "behavioral_patterns": {
                "description": "Predictable behavior patterns and tendencies",
                "subcategories": [
                    "work_patterns", "social_behavior", "adaptation_style", 
                    "stress_response", "interaction_preferences"
                ],
                "keywords": ["pattern", "behavior", "tendency", "habit", "routine"],
                "priority_weight": 0.90
            },
            "risk_factors": {
                "description": "Potential risks and vulnerabilities",
                "subcategories": [
                    "security_risks", "reputational_risks", "financial_risks", 
                    "operational_risks", "compliance_risks"
                ],
                "keywords": ["risk", "vulnerability", "threat", "danger", "exposure"],
                "priority_weight": 0.95
            },
            "career_insights": {
                "description": "Professional development and career trajectory insights",
                "subcategories": [
                    "career_progression", "skill_development", "leadership_potential", 
                    "industry_expertise", "network_strength"
                ],
                "keywords": ["career", "professional", "skill", "expertise", "development"],
                "priority_weight": 0.80
            },
            "social_dynamics": {
                "description": "Social relationships and network analysis",
                "subcategories": [
                    "network_analysis", "influence_patterns", "relationship_quality", 
                    "social_presence", "community_engagement"
                ],
                "keywords": ["social", "network", "relationship", "influence", "community"],
                "priority_weight": 0.75
            },
            "predictive_insights": {
                "description": "Future predictions and trend analysis",
                "subcategories": [
                    "career_predictions", "behavior_forecasts", "risk_projections", 
                    "opportunity_identification", "trend_analysis"
                ],
                "keywords": ["predict", "future", "forecast", "trend", "projection"],
                "priority_weight": 0.70
            },
            "anomalies": {
                "description": "Unusual patterns or inconsistencies",
                "subcategories": [
                    "data_inconsistencies", "behavioral_anomalies", "timeline_gaps", 
                    "conflicting_information", "unusual_patterns"
                ],
                "keywords": ["anomaly", "inconsistent", "unusual", "conflict", "gap"],
                "priority_weight": 0.88
            }
        }
    
    def _load_importance_weights(self) -> Dict[str, float]:
        """Load importance weights for different types of insights"""
        return {
            "high_confidence": 1.0,
            "medium_confidence": 0.7,
            "low_confidence": 0.4,
            "multiple_sources": 1.2,
            "single_source": 0.8,
            "actionable": 1.1,
            "informational": 0.9,
            "high_risk": 1.3,
            "medium_risk": 1.0,
            "low_risk": 0.7
        }
    
    async def categorize_insights(self, orchestrated_results, timeline) -> Dict[str, Any]:
        """Main method to categorize insights from all analysis results"""
        logger.info("Starting insight categorization...")
        
        # Extract insights from all sources
        all_insights = await self._extract_all_insights(orchestrated_results, timeline)
        
        # Categorize insights
        categorized_insights = await self._categorize_insights_by_type(all_insights)
        
        # Prioritize insights within categories
        prioritized_categories = await self._prioritize_insights(categorized_insights)
        
        # Generate top insights summary
        top_insights = await self._generate_top_insights(prioritized_categories)
        
        # Create actionable recommendations
        recommendations = await self._generate_recommendations(prioritized_categories)
        
        # Calculate category statistics
        category_stats = self._calculate_category_statistics(prioritized_categories)
        
        return {
            "categorized_insights": prioritized_categories,
            "top_insights": top_insights,
            "recommendations": recommendations,
            "category_statistics": category_stats,
            "metadata": {
                "total_insights": sum(len(cat.insights) for cat in prioritized_categories.values()),
                "categories_count": len(prioritized_categories),
                "high_priority_count": len([i for cat in prioritized_categories.values() 
                                          for i in cat.insights if i.importance_score > 0.8]),
                "actionable_count": len([i for cat in prioritized_categories.values() 
                                       for i in cat.insights if i.actionable]),
                "processing_timestamp": asyncio.get_event_loop().time()
            }
        }
    
    async def _extract_all_insights(self, orchestrated_results, timeline) -> List[Dict[str, Any]]:
        """Extract insights from all analysis results"""
        insights = []
        
        # Extract from personality analysis
        personality_insights = await self._extract_personality_insights(
            orchestrated_results.personality_profile
        )
        insights.extend(personality_insights)
        
        # Extract from behavioral predictions
        behavioral_insights = await self._extract_behavioral_insights(
            orchestrated_results.behavioral_predictions
        )
        insights.extend(behavioral_insights)
        
        # Extract from risk assessment
        risk_insights = await self._extract_risk_insights(
            orchestrated_results.risk_assessment
        )
        insights.extend(risk_insights)
        
        # Extract from timeline analysis
        timeline_insights = await self._extract_timeline_insights(timeline)
        insights.extend(timeline_insights)
        
        # Extract from conflicts and consensus
        consensus_insights = await self._extract_consensus_insights(
            orchestrated_results.consensus_metrics,
            orchestrated_results.conflicts_resolved
        )
        insights.extend(consensus_insights)
        
        logger.info(f"Extracted {len(insights)} total insights")
        return insights
    
    async def _extract_personality_insights(self, personality_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from personality analysis"""
        insights = []
        
        if not personality_data:
            return insights
        
        # Extract trait insights
        traits = personality_data.get("consolidated_traits", {})
        for trait, score in traits.items():
            if isinstance(score, (int, float)):
                insight_text = self._generate_trait_insight(trait, score)
                insights.append({
                    "text": insight_text,
                    "source": "personality_analysis",
                    "confidence": personality_data.get("trait_confidence", 0.7),
                    "data": {"trait": trait, "score": score},
                    "type": "personality_trait"
                })
        
        # Extract communication style insights
        comm_style = personality_data.get("communication_style", {})
        if comm_style:
            for aspect, value in comm_style.items():
                insights.append({
                    "text": f"Communication {aspect}: {value}",
                    "source": "personality_analysis",
                    "confidence": 0.75,
                    "data": {"aspect": aspect, "value": value},
                    "type": "communication_style"
                })
        
        # Extract behavioral indicators
        indicators = personality_data.get("behavioral_indicators", [])
        if indicators:
            insights.append({
                "text": f"Key behavioral indicators: {', '.join(indicators[:3])}",
                "source": "personality_analysis",
                "confidence": 0.8,
                "data": {"indicators": indicators},
                "type": "behavioral_indicators"
            })
        
        return insights
    
    async def _extract_behavioral_insights(self, behavioral_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from behavioral predictions"""
        insights = []
        
        if not behavioral_data:
            return insights
        
        # Extract decision making insights
        decision_making = behavioral_data.get("decision_making", {})
        for aspect, pattern in decision_making.items():
            insights.append({
                "text": f"Decision making - {aspect}: {pattern}",
                "source": "behavioral_prediction",
                "confidence": behavioral_data.get("prediction_confidence", 0.7),
                "data": {"aspect": aspect, "pattern": pattern},
                "type": "decision_pattern"
            })
        
        # Extract work patterns
        work_patterns = behavioral_data.get("work_patterns", {})
        for pattern, value in work_patterns.items():
            insights.append({
                "text": f"Work pattern - {pattern}: {value}",
                "source": "behavioral_prediction",
                "confidence": 0.75,
                "data": {"pattern": pattern, "value": value},
                "type": "work_pattern"
            })
        
        # Extract adaptation capacity
        adaptation = behavioral_data.get("adaptation_capacity", {})
        for metric, score in adaptation.items():
            if isinstance(score, (int, float)) and score > 0.7:
                insights.append({
                    "text": f"High {metric.replace('_', ' ')}: {score:.2f}",
                    "source": "behavioral_prediction",
                    "confidence": 0.8,
                    "data": {"metric": metric, "score": score},
                    "type": "adaptation_strength"
                })
        
        return insights
    
    async def _extract_risk_insights(self, risk_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract insights from risk assessment"""
        insights = []
        
        if not risk_data:
            return insights
        
        # Overall risk score insight
        overall_risk = risk_data.get("overall_risk_score", 0)
        if overall_risk > 0:
            risk_level = "high" if overall_risk > 7 else "medium" if overall_risk > 4 else "low"
            insights.append({
                "text": f"Overall risk level: {risk_level} (score: {overall_risk:.1f}/10)",
                "source": "risk_assessment",
                "confidence": risk_data.get("risk_confidence", 0.8),
                "data": {"overall_risk": overall_risk, "level": risk_level},
                "type": "overall_risk"
            })
        
        # Category-specific risks
        risk_categories = risk_data.get("risk_categories", {})
        for category, score in risk_categories.items():
            if isinstance(score, (int, float)) and score > 3:
                insights.append({
                    "text": f"Elevated {category.replace('_', ' ')}: {score:.1f}/10",
                    "source": "risk_assessment",
                    "confidence": 0.8,
                    "data": {"category": category, "score": score},
                    "type": "category_risk"
                })
        
        # Top risk factors
        risk_factors = risk_data.get("top_risk_factors", [])
        for factor in risk_factors[:3]:  # Top 3 risk factors
            if isinstance(factor, dict):
                insights.append({
                    "text": f"Risk factor: {factor.get('factor', 'Unknown')} - {factor.get('impact', 'medium')} impact",
                    "source": "risk_assessment",
                    "confidence": 0.85,
                    "data": factor,
                    "type": "specific_risk"
                })
        
        return insights
    
    async def _extract_timeline_insights(self, timeline) -> List[Dict[str, Any]]:
        """Extract insights from timeline analysis"""
        insights = []
        
        if not timeline or not hasattr(timeline, 'timeline_analysis'):
            return insights
        
        timeline_analysis = timeline.timeline_analysis
        
        # Timeline span and density
        span = timeline_analysis.get("timeline_span_years", 0)
        density = timeline_analysis.get("event_density_per_year", 0)
        
        if span > 0:
            insights.append({
                "text": f"Timeline spans {span} years with {density:.1f} events per year",
                "source": "timeline_analysis",
                "confidence": 0.9,
                "data": {"span": span, "density": density},
                "type": "timeline_overview"
            })
        
        # Peak activity periods
        peak_periods = timeline_analysis.get("peak_activity_periods", [])
        if peak_periods:
            years = [str(p["year"]) for p in peak_periods[:3]]
            insights.append({
                "text": f"Peak activity periods: {', '.join(years)}",
                "source": "timeline_analysis",
                "confidence": 0.8,
                "data": {"peak_periods": peak_periods},
                "type": "activity_pattern"
            })
        
        # Validation issues
        if hasattr(timeline, 'validation_result'):
            validation = timeline.validation_result
            if validation.inconsistencies:
                insights.append({
                    "text": f"Timeline inconsistencies detected: {len(validation.inconsistencies)} issues",
                    "source": "timeline_validation",
                    "confidence": 0.9,
                    "data": {"inconsistencies": validation.inconsistencies},
                    "type": "data_quality_issue"
                })
        
        return insights
    
    async def _extract_consensus_insights(self, consensus_metrics: Dict[str, Any], 
                                        conflicts: List) -> List[Dict[str, Any]]:
        """Extract insights from consensus analysis and conflicts"""
        insights = []
        
        # Overall consensus
        overall_consensus = consensus_metrics.get("overall_consensus", 0)
        if overall_consensus < 0.7:
            insights.append({
                "text": f"Low AI consensus detected: {overall_consensus:.2f} - results may need verification",
                "source": "consensus_analysis",
                "confidence": 0.9,
                "data": {"consensus_score": overall_consensus},
                "type": "data_quality_issue"
            })
        
        # Service agreement
        service_agreement = consensus_metrics.get("service_agreement", 0)
        if service_agreement > 0.8:
            insights.append({
                "text": f"High AI service agreement: {service_agreement:.2f} - results are well-supported",
                "source": "consensus_analysis",
                "confidence": 0.85,
                "data": {"agreement_score": service_agreement},
                "type": "data_quality_strength"
            })
        
        # Conflicts resolved
        if conflicts and len(conflicts) > 3:
            insights.append({
                "text": f"Multiple AI conflicts resolved: {len(conflicts)} discrepancies found and reconciled",
                "source": "conflict_resolution",
                "confidence": 0.8,
                "data": {"conflicts_count": len(conflicts)},
                "type": "data_processing_note"
            })
        
        return insights
    
    async def _categorize_insights_by_type(self, insights: List[Dict[str, Any]]) -> Dict[str, InsightCategory]:
        """Categorize insights into defined categories"""
        categorized = {}
        
        for category_name, category_config in self.categories.items():
            categorized[category_name] = InsightCategory(
                name=category_name,
                description=category_config["description"],
                insights=[],
                priority_score=0.0,
                total_insights=0,
                high_confidence_count=0
            )
        
        # Categorize each insight
        for insight_data in insights:
            category = self._determine_insight_category(insight_data)
            subcategory = self._determine_subcategory(insight_data, category)
            
            # Create CategorizedInsight object
            categorized_insight = CategorizedInsight(
                category=category,
                subcategory=subcategory,
                insight=insight_data["text"],
                importance_score=self._calculate_importance_score(insight_data, category),
                confidence=insight_data.get("confidence", 0.5),
                source_services=[insight_data.get("source", "unknown")],
                supporting_evidence=[insight_data["text"]],
                actionable=self._is_actionable(insight_data),
                risk_level=self._determine_risk_level(insight_data),
                metadata=insight_data.get("data", {})
            )
            
            # Add to appropriate category
            if category in categorized:
                categorized[category].insights.append(categorized_insight)
        
        # Update category statistics
        for category in categorized.values():
            category.total_insights = len(category.insights)
            category.high_confidence_count = len([i for i in category.insights if i.confidence > 0.8])
            if category.insights:
                category.priority_score = sum(i.importance_score for i in category.insights) / len(category.insights)
        
        return categorized
    
    def _determine_insight_category(self, insight_data: Dict[str, Any]) -> str:
        """Determine the category for an insight"""
        insight_text = insight_data["text"].lower()
        insight_type = insight_data.get("type", "").lower()
        
        # Check each category's keywords
        category_scores = {}
        for category_name, category_config in self.categories.items():
            score = 0
            keywords = category_config["keywords"]
            
            # Check keywords in text
            for keyword in keywords:
                if keyword in insight_text:
                    score += 1
            
            # Check type mapping
            if insight_type:
                if any(keyword in insight_type for keyword in keywords):
                    score += 2
            
            category_scores[category_name] = score
        
        # Return category with highest score, or default
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                return best_category[0]
        
        # Default categorization based on type
        type_mapping = {
            "personality_trait": "personality_traits",
            "behavioral_indicators": "personality_traits",
            "communication_style": "personality_traits",
            "decision_pattern": "behavioral_patterns",
            "work_pattern": "behavioral_patterns",
            "adaptation_strength": "behavioral_patterns",
            "overall_risk": "risk_factors",
            "category_risk": "risk_factors",
            "specific_risk": "risk_factors",
            "timeline_overview": "career_insights",
            "activity_pattern": "behavioral_patterns",
            "data_quality_issue": "anomalies"
        }
        
        return type_mapping.get(insight_type, "behavioral_patterns")
    
    def _determine_subcategory(self, insight_data: Dict[str, Any], category: str) -> str:
        """Determine subcategory for an insight"""
        insight_type = insight_data.get("type", "")
        
        # Mapping from insight types to subcategories
        subcategory_mapping = {
            "personality_trait": "big_five_traits",
            "communication_style": "communication_style",
            "behavioral_indicators": "decision_making",
            "decision_pattern": "decision_making",
            "work_pattern": "work_patterns",
            "adaptation_strength": "adaptation_style",
            "overall_risk": "security_risks",
            "category_risk": "security_risks",
            "specific_risk": "security_risks",
            "timeline_overview": "career_progression",
            "activity_pattern": "work_patterns",
            "data_quality_issue": "data_inconsistencies"
        }
        
        subcategory = subcategory_mapping.get(insight_type, "")
        
        # If no mapping found, use first subcategory of the category
        if not subcategory and category in self.categories:
            subcategories = self.categories[category].get("subcategories", [])
            subcategory = subcategories[0] if subcategories else "general"
        
        return subcategory or "general"
    
    def _calculate_importance_score(self, insight_data: Dict[str, Any], category: str) -> float:
        """Calculate importance score for an insight"""
        base_score = 0.5
        
        # Category weight
        category_weight = self.categories.get(category, {}).get("priority_weight", 0.5)
        base_score *= category_weight
        
        # Confidence weight
        confidence = insight_data.get("confidence", 0.5)
        if confidence > 0.8:
            base_score *= self.importance_weights["high_confidence"]
        elif confidence > 0.6:
            base_score *= self.importance_weights["medium_confidence"]
        else:
            base_score *= self.importance_weights["low_confidence"]
        
        # Risk level weight
        risk_indicators = ["risk", "threat", "vulnerability", "danger"]
        if any(indicator in insight_data["text"].lower() for indicator in risk_indicators):
            base_score *= self.importance_weights["high_risk"]
        
        # Actionability weight
        if self._is_actionable(insight_data):
            base_score *= self.importance_weights["actionable"]
        else:
            base_score *= self.importance_weights["informational"]
        
        return min(base_score, 1.0)  # Cap at 1.0
    
    def _is_actionable(self, insight_data: Dict[str, Any]) -> bool:
        """Determine if an insight is actionable"""
        actionable_keywords = [
            "recommend", "should", "consider", "improve", "develop", 
            "address", "mitigate", "enhance", "focus", "avoid"
        ]
        
        insight_text = insight_data["text"].lower()
        return any(keyword in insight_text for keyword in actionable_keywords)
    
    def _determine_risk_level(self, insight_data: Dict[str, Any]) -> str:
        """Determine risk level of an insight"""
        insight_text = insight_data["text"].lower()
        
        high_risk_keywords = ["critical", "severe", "major", "significant threat"]
        medium_risk_keywords = ["moderate", "elevated", "concerning", "attention"]
        
        if any(keyword in insight_text for keyword in high_risk_keywords):
            return "high"
        elif any(keyword in insight_text for keyword in medium_risk_keywords):
            return "medium"
        else:
            return "low"
    
    async def _prioritize_insights(self, categorized_insights: Dict[str, InsightCategory]) -> Dict[str, InsightCategory]:
        """Prioritize insights within each category"""
        for category in categorized_insights.values():
            # Sort insights by importance score (descending)
            category.insights.sort(key=lambda x: x.importance_score, reverse=True)
        
        return categorized_insights
    
    async def _generate_top_insights(self, categorized_insights: Dict[str, InsightCategory]) -> List[Dict[str, Any]]:
        """Generate top insights across all categories"""
        all_insights = []
        
        for category_name, category in categorized_insights.items():
            for insight in category.insights:
                all_insights.append({
                    "category": category_name,
                    "subcategory": insight.subcategory,
                    "insight": insight.insight,
                    "importance_score": insight.importance_score,
                    "confidence": insight.confidence,
                    "risk_level": insight.risk_level,
                    "actionable": insight.actionable
                })
        
        # Sort by importance score and return top 10
        all_insights.sort(key=lambda x: x["importance_score"], reverse=True)
        return all_insights[:10]
    
    async def _generate_recommendations(self, categorized_insights: Dict[str, InsightCategory]) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on insights"""
        recommendations = []
        
        # High-priority actionable insights
        actionable_insights = []
        for category in categorized_insights.values():
            for insight in category.insights:
                if insight.actionable and insight.importance_score > 0.7:
                    actionable_insights.append(insight)
        
        # Sort by importance
        actionable_insights.sort(key=lambda x: x.importance_score, reverse=True)
        
        # Generate recommendations
        for insight in actionable_insights[:5]:  # Top 5 actionable insights
            recommendation = self._generate_recommendation_from_insight(insight)
            if recommendation:
                recommendations.append(recommendation)
        
        # Add category-specific recommendations
        category_recommendations = self._generate_category_recommendations(categorized_insights)
        recommendations.extend(category_recommendations)
        
        return recommendations[:10]  # Return top 10 recommendations
    
    def _generate_recommendation_from_insight(self, insight: CategorizedInsight) -> Optional[Dict[str, Any]]:
        """Generate a specific recommendation from an insight"""
        if insight.category == "risk_factors":
            return {
                "type": "risk_mitigation",
                "priority": "high" if insight.risk_level == "high" else "medium",
                "recommendation": f"Address identified risk: {insight.insight}",
                "category": insight.category,
                "confidence": insight.confidence
            }
        elif insight.category == "behavioral_patterns":
            return {
                "type": "behavioral_optimization",
                "priority": "medium",
                "recommendation": f"Leverage behavioral strength: {insight.insight}",
                "category": insight.category,
                "confidence": insight.confidence
            }
        elif insight.category == "career_insights":
            return {
                "type": "career_development",
                "priority": "medium",
                "recommendation": f"Career development opportunity: {insight.insight}",
                "category": insight.category,
                "confidence": insight.confidence
            }
        
        return None
    
    def _generate_category_recommendations(self, categorized_insights: Dict[str, InsightCategory]) -> List[Dict[str, Any]]:
        """Generate recommendations based on category analysis"""
        recommendations = []
        
        # Check for categories with low insight counts
        for category_name, category in categorized_insights.items():
            if category.total_insights < 2:
                recommendations.append({
                    "type": "data_collection",
                    "priority": "low",
                    "recommendation": f"Gather more data for {category_name.replace('_', ' ')} analysis",
                    "category": category_name,
                    "confidence": 0.6
                })
        
        return recommendations
    
    def _calculate_category_statistics(self, categorized_insights: Dict[str, InsightCategory]) -> Dict[str, Any]:
        """Calculate statistics for categorized insights"""
        stats = {}
        
        total_insights = sum(cat.total_insights for cat in categorized_insights.values())
        
        for category_name, category in categorized_insights.items():
            stats[category_name] = {
                "total_insights": category.total_insights,
                "high_confidence_count": category.high_confidence_count,
                "priority_score": category.priority_score,
                "percentage_of_total": (category.total_insights / total_insights * 100) if total_insights > 0 else 0,
                "actionable_count": len([i for i in category.insights if i.actionable]),
                "high_risk_count": len([i for i in category.insights if i.risk_level == "high"])
            }
        
        return stats
    
    def _generate_trait_insight(self, trait: str, score: float) -> str:
        """Generate human-readable insight for personality traits"""
        trait_descriptions = {
            "openness": {
                "high": "highly creative and open to new experiences",
                "medium": "moderately open to new ideas and experiences", 
                "low": "prefers familiar situations and conventional approaches"
            },
            "conscientiousness": {
                "high": "highly organized and goal-oriented",
                "medium": "reasonably organized with good self-discipline",
                "low": "more flexible and spontaneous in approach"
            },
            "extraversion": {
                "high": "highly social and energetic",
                "medium": "balanced between social and solitary activities",
                "low": "prefers quieter, more reflective environments"
            },
            "agreeableness": {
                "high": "highly cooperative and trusting",
                "medium": "generally cooperative with balanced skepticism",
                "low": "more competitive and skeptical in nature"
            },
            "neuroticism": {
                "high": "tends to experience stress and emotional volatility",
                "medium": "generally emotionally stable with occasional stress",
                "low": "highly emotionally stable and resilient"
            }
        }
        
        # Determine level
        if score > 0.7:
            level = "high"
        elif score > 0.4:
            level = "medium"
        else:
            level = "low"
        
        trait_key = trait.lower()
        if trait_key in trait_descriptions:
            description = trait_descriptions[trait_key][level]
            return f"{trait.title()}: {description} (score: {score:.2f})"
        else:
            return f"{trait.title()}: {score:.2f}"
