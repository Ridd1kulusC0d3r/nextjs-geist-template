"""
Timeline Validator
Validates and creates robust timelines from orchestrated results
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class TimelineEvent:
    """Represents a single event in the timeline"""
    date: str
    event_type: str
    description: str
    source: str
    confidence: float
    impact_level: str
    category: str
    metadata: Dict[str, Any]

@dataclass
class ValidationResult:
    """Result of timeline validation"""
    is_valid: bool
    confidence_score: float
    inconsistencies: List[str]
    gaps_identified: List[str]
    recommendations: List[str]

@dataclass
class ValidatedTimeline:
    """Complete validated timeline with events and analysis"""
    events: List[TimelineEvent]
    validation_result: ValidationResult
    timeline_analysis: Dict[str, Any]
    patterns_identified: Dict[str, Any]
    metadata: Dict[str, Any]

class TimelineValidator:
    """Validates and creates robust timelines from investigation data"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.validation_rules = self._load_validation_rules()
        
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load timeline validation rules"""
        return {
            "date_format_patterns": [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{4}-\d{2}',        # YYYY-MM
                r'\d{4}'               # YYYY
            ],
            "minimum_confidence": 0.3,
            "maximum_gap_years": 2,
            "required_categories": ["education", "career", "personal"],
            "impact_levels": ["low", "medium", "high", "critical"],
            "event_types": [
                "education", "employment", "achievement", "transition",
                "certification", "publication", "award", "milestone"
            ]
        }
    
    async def create_timeline(self, orchestrated_results) -> ValidatedTimeline:
        """Create and validate timeline from orchestrated results"""
        logger.info("Creating validated timeline...")
        
        # Extract timeline events from orchestrated results
        raw_events = await self._extract_timeline_events(orchestrated_results)
        
        # Validate and clean events
        validated_events = await self._validate_events(raw_events)
        
        # Sort events chronologically
        sorted_events = self._sort_events_chronologically(validated_events)
        
        # Identify and fill gaps
        gap_filled_events = await self._identify_and_fill_gaps(sorted_events)
        
        # Validate overall timeline consistency
        validation_result = await self._validate_timeline_consistency(gap_filled_events)
        
        # Analyze timeline patterns
        timeline_analysis = await self._analyze_timeline_patterns(gap_filled_events)
        
        # Identify behavioral and career patterns
        patterns_identified = await self._identify_patterns(gap_filled_events, orchestrated_results)
        
        return ValidatedTimeline(
            events=gap_filled_events,
            validation_result=validation_result,
            timeline_analysis=timeline_analysis,
            patterns_identified=patterns_identified,
            metadata={
                "creation_timestamp": datetime.now().isoformat(),
                "total_events": len(gap_filled_events),
                "validation_score": validation_result.confidence_score,
                "timeline_span": self._calculate_timeline_span(gap_filled_events),
                "data_sources": self._get_data_sources(gap_filled_events)
            }
        )
    
    async def _extract_timeline_events(self, orchestrated_results) -> List[TimelineEvent]:
        """Extract timeline events from orchestrated results"""
        events = []
        
        # Extract from timeline analysis
        timeline_data = orchestrated_results.timeline_analysis
        if timeline_data.get("key_milestones"):
            for milestone in timeline_data["key_milestones"]:
                event = self._create_event_from_milestone(milestone, "timeline_analysis")
                if event:
                    events.append(event)
        
        # Extract from career progression data
        if timeline_data.get("career_progression"):
            career_events = self._extract_career_events(timeline_data, "career_analysis")
            events.extend(career_events)
        
        # Extract from behavioral predictions (for future events)
        behavioral_data = orchestrated_results.behavioral_predictions
        if behavioral_data.get("future_projections"):
            future_events = self._extract_future_events(behavioral_data, "behavioral_prediction")
            events.extend(future_events)
        
        # Extract from risk assessment (for risk-related timeline events)
        risk_data = orchestrated_results.risk_assessment
        if risk_data.get("top_risk_factors"):
            risk_events = self._extract_risk_timeline_events(risk_data, "risk_assessment")
            events.extend(risk_events)
        
        logger.info(f"Extracted {len(events)} timeline events")
        return events
    
    def _create_event_from_milestone(self, milestone: Dict[str, Any], source: str) -> Optional[TimelineEvent]:
        """Create timeline event from milestone data"""
        try:
            # Handle different milestone formats
            date = milestone.get("date") or milestone.get("start_date") or milestone.get("graduation_year")
            if not date:
                return None
            
            event_type = milestone.get("event") or milestone.get("phase") or "milestone"
            description = milestone.get("description") or milestone.get("event") or milestone.get("phase", "")
            
            # Determine impact level
            impact_level = "medium"
            if milestone.get("significance") == "high":
                impact_level = "high"
            elif "graduation" in description.lower() or "certification" in description.lower():
                impact_level = "high"
            
            # Determine category
            category = self._determine_event_category(description, event_type)
            
            return TimelineEvent(
                date=str(date),
                event_type=event_type,
                description=description,
                source=source,
                confidence=milestone.get("confidence", 0.7),
                impact_level=impact_level,
                category=category,
                metadata=milestone
            )
        except Exception as e:
            logger.warning(f"Failed to create event from milestone: {e}")
            return None
    
    def _extract_career_events(self, timeline_data: Dict[str, Any], source: str) -> List[TimelineEvent]:
        """Extract career-related events"""
        events = []
        
        # Extract from career progression
        progression = timeline_data.get("career_progression", "")
        if progression:
            # Create a general career progression event
            events.append(TimelineEvent(
                date="2020",  # Approximate date
                event_type="career_progression",
                description=f"Career progression pattern: {progression}",
                source=source,
                confidence=0.6,
                impact_level="medium",
                category="career",
                metadata={"progression_type": progression}
            ))
        
        return events
    
    def _extract_future_events(self, behavioral_data: Dict[str, Any], source: str) -> List[TimelineEvent]:
        """Extract predicted future events"""
        events = []
        
        projections = behavioral_data.get("future_projections", {})
        current_year = datetime.now().year
        
        # Extract 5-year outlook
        if "5_year_outlook" in projections:
            events.append(TimelineEvent(
                date=str(current_year + 5),
                event_type="predicted_milestone",
                description=f"Predicted: {projections['5_year_outlook']}",
                source=source,
                confidence=projections.get("probability_confidence", 0.5),
                impact_level="high",
                category="career",
                metadata={"prediction_type": "5_year", "original_data": projections}
            ))
        
        # Extract 10-year outlook
        if "10_year_outlook" in projections:
            events.append(TimelineEvent(
                date=str(current_year + 10),
                event_type="predicted_milestone",
                description=f"Predicted: {projections['10_year_outlook']}",
                source=source,
                confidence=projections.get("probability_confidence", 0.4),
                impact_level="high",
                category="career",
                metadata={"prediction_type": "10_year", "original_data": projections}
            ))
        
        return events
    
    def _extract_risk_timeline_events(self, risk_data: Dict[str, Any], source: str) -> List[TimelineEvent]:
        """Extract risk-related timeline events"""
        events = []
        
        for risk_factor in risk_data.get("top_risk_factors", []):
            if isinstance(risk_factor, dict):
                events.append(TimelineEvent(
                    date=str(datetime.now().year),  # Current year for risk factors
                    event_type="risk_factor",
                    description=f"Risk identified: {risk_factor.get('factor', 'Unknown risk')}",
                    source=source,
                    confidence=0.8,
                    impact_level=risk_factor.get("impact", "medium"),
                    category="risk",
                    metadata=risk_factor
                ))
        
        return events
    
    async def _validate_events(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Validate individual timeline events"""
        validated_events = []
        
        for event in events:
            if await self._is_valid_event(event):
                # Normalize the event
                normalized_event = self._normalize_event(event)
                validated_events.append(normalized_event)
            else:
                logger.warning(f"Invalid event filtered out: {event.description}")
        
        logger.info(f"Validated {len(validated_events)} out of {len(events)} events")
        return validated_events
    
    async def _is_valid_event(self, event: TimelineEvent) -> bool:
        """Check if an event is valid"""
        # Check date format
        if not self._is_valid_date(event.date):
            return False
        
        # Check confidence threshold
        if event.confidence < self.validation_rules["minimum_confidence"]:
            return False
        
        # Check required fields
        if not event.description or not event.event_type:
            return False
        
        # Check impact level
        if event.impact_level not in self.validation_rules["impact_levels"]:
            return False
        
        return True
    
    def _is_valid_date(self, date_str: str) -> bool:
        """Validate date format"""
        for pattern in self.validation_rules["date_format_patterns"]:
            if re.match(pattern, date_str):
                return True
        return False
    
    def _normalize_event(self, event: TimelineEvent) -> TimelineEvent:
        """Normalize event data"""
        # Normalize date to YYYY-MM-DD format where possible
        normalized_date = self._normalize_date(event.date)
        
        # Normalize event type
        normalized_type = event.event_type.lower().replace(" ", "_")
        
        # Ensure category is valid
        normalized_category = event.category if event.category in self.validation_rules["required_categories"] else "other"
        
        return TimelineEvent(
            date=normalized_date,
            event_type=normalized_type,
            description=event.description.strip(),
            source=event.source,
            confidence=min(max(event.confidence, 0.0), 1.0),  # Clamp between 0 and 1
            impact_level=event.impact_level,
            category=normalized_category,
            metadata=event.metadata
        )
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to consistent format"""
        # Try to parse and reformat common date formats
        try:
            # Handle YYYY format
            if re.match(r'^\d{4}$', date_str):
                return f"{date_str}-01-01"
            
            # Handle YYYY-MM format
            if re.match(r'^\d{4}-\d{2}$', date_str):
                return f"{date_str}-01"
            
            # Handle MM/DD/YYYY format
            mm_dd_yyyy = re.match(r'^(\d{2})/(\d{2})/(\d{4})$', date_str)
            if mm_dd_yyyy:
                month, day, year = mm_dd_yyyy.groups()
                return f"{year}-{month}-{day}"
            
            # Already in YYYY-MM-DD format
            if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
                return date_str
            
        except Exception as e:
            logger.warning(f"Date normalization failed for {date_str}: {e}")
        
        return date_str  # Return original if normalization fails
    
    def _sort_events_chronologically(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Sort events in chronological order"""
        def date_sort_key(event):
            try:
                # Extract year for sorting
                year_match = re.search(r'(\d{4})', event.date)
                if year_match:
                    return int(year_match.group(1))
                return 0
            except:
                return 0
        
        return sorted(events, key=date_sort_key)
    
    async def _identify_and_fill_gaps(self, events: List[TimelineEvent]) -> List[TimelineEvent]:
        """Identify timeline gaps and attempt to fill them"""
        if len(events) < 2:
            return events
        
        filled_events = events.copy()
        gaps_identified = []
        
        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]
            
            gap_years = self._calculate_year_gap(current_event.date, next_event.date)
            
            if gap_years > self.validation_rules["maximum_gap_years"]:
                # Identify gap
                gap_info = f"Gap of {gap_years} years between {current_event.date} and {next_event.date}"
                gaps_identified.append(gap_info)
                logger.info(f"Timeline gap identified: {gap_info}")
                
                # Attempt to fill gap with inferred events
                inferred_events = self._infer_gap_events(current_event, next_event, gap_years)
                filled_events.extend(inferred_events)
        
        # Re-sort after adding inferred events
        return self._sort_events_chronologically(filled_events)
    
    def _calculate_year_gap(self, date1: str, date2: str) -> int:
        """Calculate year gap between two dates"""
        try:
            year1 = int(re.search(r'(\d{4})', date1).group(1))
            year2 = int(re.search(r'(\d{4})', date2).group(1))
            return abs(year2 - year1)
        except:
            return 0
    
    def _infer_gap_events(self, event1: TimelineEvent, event2: TimelineEvent, gap_years: int) -> List[TimelineEvent]:
        """Infer events to fill timeline gaps"""
        inferred_events = []
        
        try:
            start_year = int(re.search(r'(\d{4})', event1.date).group(1))
            end_year = int(re.search(r'(\d{4})', event2.date).group(1))
            
            # Create intermediate milestone events
            for year in range(start_year + 1, end_year):
                if (year - start_year) % 2 == 0:  # Every 2 years
                    inferred_events.append(TimelineEvent(
                        date=f"{year}-01-01",
                        event_type="inferred_milestone",
                        description=f"Inferred activity period (between {event1.description} and {event2.description})",
                        source="timeline_inference",
                        confidence=0.3,
                        impact_level="low",
                        category="inferred",
                        metadata={
                            "inferred": True,
                            "between_events": [event1.description, event2.description]
                        }
                    ))
        except Exception as e:
            logger.warning(f"Failed to infer gap events: {e}")
        
        return inferred_events
    
    async def _validate_timeline_consistency(self, events: List[TimelineEvent]) -> ValidationResult:
        """Validate overall timeline consistency"""
        inconsistencies = []
        gaps_identified = []
        recommendations = []
        
        # Check for chronological consistency
        for i in range(len(events) - 1):
            current_year = self._extract_year(events[i].date)
            next_year = self._extract_year(events[i + 1].date)
            
            if current_year and next_year and current_year > next_year:
                inconsistencies.append(f"Chronological inconsistency: {events[i].description} ({current_year}) after {events[i + 1].description} ({next_year})")
        
        # Check for major gaps
        for i in range(len(events) - 1):
            gap = self._calculate_year_gap(events[i].date, events[i + 1].date)
            if gap > self.validation_rules["maximum_gap_years"]:
                gaps_identified.append(f"Major gap: {gap} years between {events[i].description} and {events[i + 1].description}")
        
        # Check category coverage
        categories_present = set(event.category for event in events)
        missing_categories = set(self.validation_rules["required_categories"]) - categories_present
        if missing_categories:
            recommendations.append(f"Consider adding events in categories: {', '.join(missing_categories)}")
        
        # Check confidence levels
        low_confidence_events = [e for e in events if e.confidence < 0.5]
        if low_confidence_events:
            recommendations.append(f"Verify {len(low_confidence_events)} low-confidence events")
        
        # Calculate overall confidence score
        if events:
            avg_confidence = sum(event.confidence for event in events) / len(events)
            consistency_penalty = len(inconsistencies) * 0.1
            gap_penalty = len(gaps_identified) * 0.05
            confidence_score = max(0.0, avg_confidence - consistency_penalty - gap_penalty)
        else:
            confidence_score = 0.0
        
        is_valid = len(inconsistencies) == 0 and confidence_score > 0.5
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_score=confidence_score,
            inconsistencies=inconsistencies,
            gaps_identified=gaps_identified,
            recommendations=recommendations
        )
    
    async def _analyze_timeline_patterns(self, events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze patterns in the timeline"""
        if not events:
            return {}
        
        # Analyze event frequency by category
        category_counts = defaultdict(int)
        for event in events:
            category_counts[event.category] += 1
        
        # Analyze timeline density (events per year)
        timeline_span = self._calculate_timeline_span(events)
        event_density = len(events) / max(timeline_span, 1)
        
        # Analyze confidence trends
        confidence_trend = self._analyze_confidence_trend(events)
        
        # Analyze impact distribution
        impact_distribution = defaultdict(int)
        for event in events:
            impact_distribution[event.impact_level] += 1
        
        # Identify peak activity periods
        peak_periods = self._identify_peak_periods(events)
        
        return {
            "timeline_span_years": timeline_span,
            "total_events": len(events),
            "event_density_per_year": round(event_density, 2),
            "category_distribution": dict(category_counts),
            "impact_distribution": dict(impact_distribution),
            "confidence_trend": confidence_trend,
            "peak_activity_periods": peak_periods,
            "average_confidence": sum(e.confidence for e in events) / len(events),
            "data_source_diversity": len(set(e.source for e in events))
        }
    
    async def _identify_patterns(self, events: List[TimelineEvent], orchestrated_results) -> Dict[str, Any]:
        """Identify behavioral and career patterns from timeline"""
        patterns = {}
        
        # Career progression patterns
        career_events = [e for e in events if e.category == "career"]
        if career_events:
            patterns["career_progression"] = self._analyze_career_progression(career_events)
        
        # Education patterns
        education_events = [e for e in events if e.category == "education"]
        if education_events:
            patterns["education_pattern"] = self._analyze_education_pattern(education_events)
        
        # Achievement patterns
        achievement_events = [e for e in events if "achievement" in e.event_type or e.impact_level == "high"]
        if achievement_events:
            patterns["achievement_pattern"] = self._analyze_achievement_pattern(achievement_events)
        
        # Risk patterns
        risk_events = [e for e in events if e.category == "risk"]
        if risk_events:
            patterns["risk_pattern"] = self._analyze_risk_pattern(risk_events)
        
        # Behavioral consistency
        patterns["behavioral_consistency"] = self._analyze_behavioral_consistency(events, orchestrated_results)
        
        return patterns
    
    # Helper methods
    def _determine_event_category(self, description: str, event_type: str) -> str:
        """Determine event category based on description and type"""
        description_lower = description.lower()
        event_type_lower = event_type.lower()
        
        if any(keyword in description_lower for keyword in ["education", "graduation", "degree", "school", "university"]):
            return "education"
        elif any(keyword in description_lower for keyword in ["job", "career", "work", "employment", "company"]):
            return "career"
        elif any(keyword in description_lower for keyword in ["risk", "threat", "vulnerability"]):
            return "risk"
        elif any(keyword in event_type_lower for keyword in ["achievement", "award", "certification"]):
            return "achievement"
        else:
            return "other"
    
    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string"""
        try:
            year_match = re.search(r'(\d{4})', date_str)
            return int(year_match.group(1)) if year_match else None
        except:
            return None
    
    def _calculate_timeline_span(self, events: List[TimelineEvent]) -> int:
        """Calculate timeline span in years"""
        if len(events) < 2:
            return 1
        
        years = [self._extract_year(event.date) for event in events]
        valid_years = [year for year in years if year is not None]
        
        if len(valid_years) < 2:
            return 1
        
        return max(valid_years) - min(valid_years)
    
    def _get_data_sources(self, events: List[TimelineEvent]) -> List[str]:
        """Get unique data sources from events"""
        return list(set(event.source for event in events))
    
    def _analyze_confidence_trend(self, events: List[TimelineEvent]) -> str:
        """Analyze confidence trend over time"""
        if len(events) < 3:
            return "insufficient_data"
        
        # Sort by date and analyze confidence progression
        sorted_events = self._sort_events_chronologically(events)
        confidences = [event.confidence for event in sorted_events]
        
        # Simple trend analysis
        first_half_avg = sum(confidences[:len(confidences)//2]) / (len(confidences)//2)
        second_half_avg = sum(confidences[len(confidences)//2:]) / (len(confidences) - len(confidences)//2)
        
        if second_half_avg > first_half_avg + 0.1:
            return "improving"
        elif second_half_avg < first_half_avg - 0.1:
            return "declining"
        else:
            return "stable"
    
    def _identify_peak_periods(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Identify periods of high activity"""
        if not events:
            return []
        
        # Group events by year
        year_counts = defaultdict(int)
        for event in events:
            year = self._extract_year(event.date)
            if year:
                year_counts[year] += 1
        
        # Find years with above-average activity
        if not year_counts:
            return []
        
        avg_activity = sum(year_counts.values()) / len(year_counts)
        peak_years = [year for year, count in year_counts.items() if count > avg_activity * 1.5]
        
        return [{"year": year, "event_count": year_counts[year]} for year in sorted(peak_years)]
    
    def _analyze_career_progression(self, career_events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze career progression patterns"""
        if not career_events:
            return {}
        
        return {
            "total_career_events": len(career_events),
            "career_span_years": self._calculate_timeline_span(career_events),
            "progression_rate": len(career_events) / max(self._calculate_timeline_span(career_events), 1),
            "average_confidence": sum(e.confidence for e in career_events) / len(career_events)
        }
    
    def _analyze_education_pattern(self, education_events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze education patterns"""
        if not education_events:
            return {}
        
        return {
            "total_education_events": len(education_events),
            "education_span_years": self._calculate_timeline_span(education_events),
            "continuous_learning": len(education_events) > 2,
            "average_confidence": sum(e.confidence for e in education_events) / len(education_events)
        }
    
    def _analyze_achievement_pattern(self, achievement_events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze achievement patterns"""
        if not achievement_events:
            return {}
        
        return {
            "total_achievements": len(achievement_events),
            "achievement_frequency": len(achievement_events) / max(self._calculate_timeline_span(achievement_events), 1),
            "high_impact_achievements": len([e for e in achievement_events if e.impact_level == "high"]),
            "average_confidence": sum(e.confidence for e in achievement_events) / len(achievement_events)
        }
    
    def _analyze_risk_pattern(self, risk_events: List[TimelineEvent]) -> Dict[str, Any]:
        """Analyze risk patterns"""
        if not risk_events:
            return {}
        
        return {
            "total_risk_events": len(risk_events),
            "risk_distribution": {level: len([e for e in risk_events if e.impact_level == level]) 
                               for level in ["low", "medium", "high"]},
            "average_confidence": sum(e.confidence for e in risk_events) / len(risk_events)
        }
    
    def _analyze_behavioral_consistency(self, events: List[TimelineEvent], orchestrated_results) -> Dict[str, Any]:
        """Analyze behavioral consistency across timeline"""
        # This would correlate timeline events with behavioral predictions
        # For now, return basic consistency metrics
        
        confidence_variance = 0
        if len(events) > 1:
            confidences = [e.confidence for e in events]
            mean_confidence = sum(confidences) / len(confidences)
            confidence_variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        
        return {
            "confidence_consistency": 1.0 - min(confidence_variance, 1.0),
            "source_diversity": len(set(e.source for e in events)),
            "category_balance": len(set(e.category for e in events)) / max(len(self.validation_rules["required_categories"]), 1)
        }
