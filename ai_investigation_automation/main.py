"""
AI Investigation Automation System
Main orchestration script for automated investigation and analysis
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from dataclasses import dataclass

from data_collector import DataCollector
from ai_services import AIServiceManager
from orchestrator import ResultOrchestrator
from timeline_validator import TimelineValidator
from insight_categorizer import InsightCategorizer
from report_generator import ReportGenerator

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class InvestigationTarget:
    """Data class for investigation target information"""
    name: str
    email: str
    additional_data: Dict[str, Any]

class AIInvestigationSystem:
    """Main system class for AI-powered investigation automation"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the investigation system with configuration"""
        self.config = self._load_config(config_path)
        self.data_collector = DataCollector(self.config)
        self.ai_service_manager = AIServiceManager(self.config)
        self.orchestrator = ResultOrchestrator(self.config)
        self.timeline_validator = TimelineValidator(self.config)
        self.insight_categorizer = InsightCategorizer(self.config)
        self.report_generator = ReportGenerator(self.config)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found. Using default configuration.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "ai_services": {
                "openai": {"enabled": True, "api_key": ""},
                "anthropic": {"enabled": True, "api_key": ""},
                "google": {"enabled": True, "api_key": ""}
            },
            "data_sources": {
                "social_media": True,
                "public_records": True,
                "professional_networks": True
            },
            "analysis_types": {
                "personality": True,
                "behavior_prediction": True,
                "risk_assessment": True,
                "timeline_analysis": True
            }
        }
    
    async def run_investigation(self, target: InvestigationTarget) -> Dict[str, Any]:
        """Run complete investigation workflow"""
        logger.info(f"Starting investigation for: {target.name}")
        
        try:
            # Step 1: Collect initial data
            logger.info("Step 1: Collecting data...")
            collected_data = await self.data_collector.collect_data(target)
            
            # Step 2: Run AI analyses
            logger.info("Step 2: Running AI analyses...")
            ai_results = await self.ai_service_manager.run_analyses(collected_data)
            
            # Step 3: Orchestrate and merge results
            logger.info("Step 3: Orchestrating results...")
            orchestrated_results = await self.orchestrator.orchestrate_results(ai_results)
            
            # Step 4: Validate and create timeline
            logger.info("Step 4: Creating validated timeline...")
            timeline = await self.timeline_validator.create_timeline(orchestrated_results)
            
            # Step 5: Categorize insights
            logger.info("Step 5: Categorizing insights...")
            categorized_insights = await self.insight_categorizer.categorize_insights(
                orchestrated_results, timeline
            )
            
            # Step 6: Generate report
            logger.info("Step 6: Generating report...")
            report = await self.report_generator.generate_report(
                target, orchestrated_results, timeline, categorized_insights
            )
            
            logger.info("Investigation completed successfully!")
            return {
                "target": target,
                "data": collected_data,
                "ai_results": ai_results,
                "orchestrated_results": orchestrated_results,
                "timeline": timeline,
                "insights": categorized_insights,
                "report": report,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Investigation failed: {str(e)}")
            raise

async def main():
    """Main function for running the investigation system"""
    # Example usage
    target = InvestigationTarget(
        name="John Doe",
        email="john.doe@example.com",
        additional_data={
            "phone": "+1234567890",
            "location": "New York, NY",
            "company": "Tech Corp"
        }
    )
    
    system = AIInvestigationSystem()
    results = await system.run_investigation(target)
    
    print("Investigation Results:")
    print(f"Target: {results['target'].name}")
    print(f"Report generated at: {results['timestamp']}")
    print(f"Report path: {results['report']['file_path']}")

if __name__ == "__main__":
    asyncio.run(main())
