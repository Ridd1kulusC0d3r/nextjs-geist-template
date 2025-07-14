"""
Data Collector Module
Handles collection of relevant data about investigation targets
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import json
import re

logger = logging.getLogger(__name__)

@dataclass
class CollectedData:
    """Data structure for collected information"""
    basic_info: Dict[str, Any]
    social_media: Dict[str, Any]
    public_records: Dict[str, Any]
    professional_info: Dict[str, Any]
    digital_footprint: Dict[str, Any]
    metadata: Dict[str, Any]

class DataCollector:
    """Collects data from various sources about investigation targets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def collect_data(self, target) -> CollectedData:
        """Main method to collect all available data about target"""
        logger.info(f"Starting data collection for {target.name}")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            # Collect data from different sources concurrently
            tasks = []
            
            if self.config.get("data_sources", {}).get("social_media", True):
                tasks.append(self._collect_social_media_data(target))
            
            if self.config.get("data_sources", {}).get("public_records", True):
                tasks.append(self._collect_public_records(target))
                
            if self.config.get("data_sources", {}).get("professional_networks", True):
                tasks.append(self._collect_professional_data(target))
            
            # Execute all collection tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            social_media_data = results[0] if len(results) > 0 and not isinstance(results[0], Exception) else {}
            public_records_data = results[1] if len(results) > 1 and not isinstance(results[1], Exception) else {}
            professional_data = results[2] if len(results) > 2 and not isinstance(results[2], Exception) else {}
            
            # Collect digital footprint
            digital_footprint = await self._collect_digital_footprint(target)
            
            return CollectedData(
                basic_info={
                    "name": target.name,
                    "email": target.email,
                    **target.additional_data
                },
                social_media=social_media_data,
                public_records=public_records_data,
                professional_info=professional_data,
                digital_footprint=digital_footprint,
                metadata={
                    "collection_timestamp": asyncio.get_event_loop().time(),
                    "sources_accessed": self._get_accessed_sources(),
                    "data_quality_score": self._calculate_data_quality_score(
                        social_media_data, public_records_data, professional_data
                    )
                }
            )
    
    async def _collect_social_media_data(self, target) -> Dict[str, Any]:
        """Collect social media information (simulated)"""
        logger.info("Collecting social media data...")
        
        # Simulate social media data collection
        # In real implementation, this would use APIs from various platforms
        await asyncio.sleep(1)  # Simulate API call delay
        
        return {
            "platforms": {
                "linkedin": {
                    "profile_url": f"https://linkedin.com/in/{target.name.lower().replace(' ', '-')}",
                    "connections": 500,
                    "posts_count": 25,
                    "activity_level": "moderate",
                    "professional_summary": "Experienced professional in technology sector"
                },
                "twitter": {
                    "handle": f"@{target.name.lower().replace(' ', '')}",
                    "followers": 1200,
                    "following": 800,
                    "tweets_count": 3400,
                    "sentiment": "neutral",
                    "topics": ["technology", "business", "innovation"]
                },
                "facebook": {
                    "privacy_level": "high",
                    "public_posts": 5,
                    "friends_visible": False,
                    "location_sharing": "limited"
                }
            },
            "analysis": {
                "social_presence_score": 7.5,
                "engagement_rate": "medium",
                "content_themes": ["professional", "technology", "personal interests"],
                "posting_frequency": "weekly"
            }
        }
    
    async def _collect_public_records(self, target) -> Dict[str, Any]:
        """Collect public records information (simulated)"""
        logger.info("Collecting public records...")
        
        # Simulate public records search
        await asyncio.sleep(1.5)  # Simulate search delay
        
        return {
            "business_registrations": [
                {
                    "company_name": "Tech Solutions LLC",
                    "role": "Director",
                    "registration_date": "2020-03-15",
                    "status": "active"
                }
            ],
            "property_records": [
                {
                    "address": "123 Main St, New York, NY",
                    "ownership_type": "primary_residence",
                    "purchase_date": "2019-08-20",
                    "estimated_value": 750000
                }
            ],
            "court_records": [],
            "professional_licenses": [
                {
                    "license_type": "Professional Engineer",
                    "license_number": "PE123456",
                    "issue_date": "2018-06-01",
                    "expiry_date": "2024-06-01",
                    "status": "active"
                }
            ],
            "education_records": [
                {
                    "institution": "MIT",
                    "degree": "Master of Science",
                    "field": "Computer Science",
                    "graduation_year": "2015"
                }
            ]
        }
    
    async def _collect_professional_data(self, target) -> Dict[str, Any]:
        """Collect professional network information (simulated)"""
        logger.info("Collecting professional data...")
        
        await asyncio.sleep(1)  # Simulate API delay
        
        return {
            "current_employment": {
                "company": target.additional_data.get("company", "Unknown"),
                "position": "Senior Software Engineer",
                "duration": "3 years",
                "industry": "Technology"
            },
            "work_history": [
                {
                    "company": "Previous Tech Corp",
                    "position": "Software Engineer",
                    "duration": "2 years",
                    "start_date": "2018-01-01",
                    "end_date": "2020-12-31"
                }
            ],
            "skills": [
                "Python", "Machine Learning", "Cloud Computing", 
                "Project Management", "Team Leadership"
            ],
            "certifications": [
                {
                    "name": "AWS Solutions Architect",
                    "issuer": "Amazon Web Services",
                    "issue_date": "2021-09-15"
                }
            ],
            "professional_network": {
                "connections_count": 500,
                "industry_connections": 350,
                "geographic_spread": ["New York", "San Francisco", "Boston"]
            }
        }
    
    async def _collect_digital_footprint(self, target) -> Dict[str, Any]:
        """Collect digital footprint information"""
        logger.info("Analyzing digital footprint...")
        
        await asyncio.sleep(0.5)
        
        return {
            "email_analysis": {
                "domain": target.email.split('@')[1] if '@' in target.email else "",
                "email_type": "professional" if "company" in target.email else "personal",
                "breach_check": "clean",  # Simulated breach database check
                "associated_accounts": 15
            },
            "web_presence": {
                "personal_website": f"https://{target.name.lower().replace(' ', '')}.com",
                "blog_posts": 12,
                "forum_participation": "moderate",
                "online_mentions": 45
            },
            "digital_behavior": {
                "online_activity_level": "high",
                "privacy_awareness": "medium",
                "digital_security_score": 8.2,
                "technology_adoption": "early_adopter"
            }
        }
    
    def _get_accessed_sources(self) -> List[str]:
        """Get list of data sources that were accessed"""
        sources = []
        if self.config.get("data_sources", {}).get("social_media", True):
            sources.append("social_media")
        if self.config.get("data_sources", {}).get("public_records", True):
            sources.append("public_records")
        if self.config.get("data_sources", {}).get("professional_networks", True):
            sources.append("professional_networks")
        return sources
    
    def _calculate_data_quality_score(self, social_data: Dict, public_data: Dict, 
                                    professional_data: Dict) -> float:
        """Calculate overall data quality score"""
        score = 0.0
        total_sources = 0
        
        if social_data:
            score += len(social_data.get("platforms", {})) * 2
            total_sources += 1
            
        if public_data:
            score += len(public_data.get("business_registrations", [])) * 3
            score += len(public_data.get("property_records", [])) * 2
            total_sources += 1
            
        if professional_data:
            score += len(professional_data.get("skills", [])) * 0.5
            score += len(professional_data.get("certifications", [])) * 2
            total_sources += 1
        
        return min(score / max(total_sources, 1), 10.0)  # Normalize to 0-10 scale

    def validate_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def sanitize_name(self, name: str) -> str:
        """Sanitize name for safe processing"""
        return re.sub(r'[^a-zA-Z\s-]', '', name).strip()
