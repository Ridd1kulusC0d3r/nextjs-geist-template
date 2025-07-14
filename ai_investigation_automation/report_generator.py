"""
Report Generator
Generates comprehensive investigation reports from analyzed data
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ReportSection:
    """Represents a section of the investigation report"""
    title: str
    content: str
    subsections: List['ReportSection']
    metadata: Dict[str, Any]

@dataclass
class GeneratedReport:
    """Complete generated report with metadata"""
    title: str
    executive_summary: str
    sections: List[ReportSection]
    file_path: str
    format_type: str
    generation_timestamp: str
    metadata: Dict[str, Any]

class ReportGenerator:
    """Generates comprehensive investigation reports"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.report_templates = self._load_report_templates()
        self.output_dir = config.get("report_output_dir", "reports")
        self._ensure_output_directory()
        
    def _load_report_templates(self) -> Dict[str, str]:
        """Load report templates"""
        return {
            "executive_summary": """
# RELATÓRIO DE INVESTIGAÇÃO AUTOMATIZADA

## RESUMO EXECUTIVO

**Alvo da Investigação:** {target_name}
**Email:** {target_email}
**Data do Relatório:** {report_date}
**Pontuação de Confiança Geral:** {overall_confidence:.1%}

### Principais Descobertas:
{key_findings}

### Nível de Risco Geral: {risk_level}
{risk_summary}

### Recomendações Prioritárias:
{top_recommendations}
            """,
            
            "personality_section": """
## ANÁLISE DE PERSONALIDADE

### Traços Principais:
{personality_traits}

### Estilo de Comunicação:
{communication_style}

### Indicadores Comportamentais:
{behavioral_indicators}

**Confiança da Análise:** {confidence:.1%}
**Fontes:** {sources_count} serviços de IA
            """,
            
            "behavioral_section": """
## PREDIÇÕES COMPORTAMENTAIS

### Padrões de Tomada de Decisão:
{decision_patterns}

### Padrões de Trabalho:
{work_patterns}

### Capacidade de Adaptação:
{adaptation_capacity}

### Padrões de Interação:
{interaction_patterns}

**Confiança das Predições:** {confidence:.1%}
            """,
            
            "risk_section": """
## AVALIAÇÃO DE RISCOS

### Pontuação Geral de Risco: {overall_risk_score:.1f}/10

### Riscos por Categoria:
{risk_categories}

### Principais Fatores de Risco:
{top_risk_factors}

### Recomendações de Mitigação:
{risk_recommendations}

**Confiança da Avaliação:** {confidence:.1%}
            """,
            
            "timeline_section": """
## ANÁLISE DE TIMELINE

### Visão Geral:
- **Período Analisado:** {timeline_span} anos
- **Total de Eventos:** {total_events}
- **Densidade de Eventos:** {event_density:.1f} eventos/ano

### Marcos Principais:
{key_milestones}

### Períodos de Pico de Atividade:
{peak_periods}

### Padrões Identificados:
{timeline_patterns}

**Confiança da Timeline:** {confidence:.1%}
            """,
            
            "insights_section": """
## INSIGHTS CATEGORIZADOS

### Top 10 Insights:
{top_insights}

### Insights por Categoria:
{categorized_insights}

### Recomendações Acionáveis:
{actionable_recommendations}
            """,
            
            "technical_section": """
## DETALHES TÉCNICOS

### Fontes de Dados Utilizadas:
{data_sources}

### Serviços de IA Utilizados:
{ai_services}

### Métricas de Consenso:
- **Taxa de Sucesso das Análises:** {success_rate:.1%}
- **Confiança Média:** {average_confidence:.1%}
- **Concordância entre Serviços:** {service_agreement:.1%}
- **Consenso Geral:** {overall_consensus:.1%}

### Conflitos Resolvidos:
{conflicts_resolved}

### Qualidade dos Dados:
{data_quality_metrics}
            """
        }
    
    def _ensure_output_directory(self):
        """Ensure output directory exists"""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    async def generate_report(self, target, orchestrated_results, timeline, 
                            categorized_insights) -> GeneratedReport:
        """Generate comprehensive investigation report"""
        logger.info(f"Generating report for {target.name}")
        
        # Generate report sections
        sections = []
        
        # Executive Summary
        executive_summary = await self._generate_executive_summary(
            target, orchestrated_results, categorized_insights
        )
        
        # Personality Analysis Section
        personality_section = await self._generate_personality_section(
            orchestrated_results.personality_profile
        )
        sections.append(personality_section)
        
        # Behavioral Predictions Section
        behavioral_section = await self._generate_behavioral_section(
            orchestrated_results.behavioral_predictions
        )
        sections.append(behavioral_section)
        
        # Risk Assessment Section
        risk_section = await self._generate_risk_section(
            orchestrated_results.risk_assessment
        )
        sections.append(risk_section)
        
        # Timeline Analysis Section
        timeline_section = await self._generate_timeline_section(timeline)
        sections.append(timeline_section)
        
        # Insights Section
        insights_section = await self._generate_insights_section(categorized_insights)
        sections.append(insights_section)
        
        # Technical Details Section
        technical_section = await self._generate_technical_section(
            orchestrated_results, timeline, categorized_insights
        )
        sections.append(technical_section)
        
        # Generate report title
        report_title = f"Relatório de Investigação - {target.name}"
        
        # Save report to file
        file_path = await self._save_report_to_file(
            target, executive_summary, sections, report_title
        )
        
        return GeneratedReport(
            title=report_title,
            executive_summary=executive_summary,
            sections=sections,
            file_path=file_path,
            format_type="markdown",
            generation_timestamp=datetime.now().isoformat(),
            metadata={
                "target_name": target.name,
                "target_email": target.email,
                "sections_count": len(sections),
                "total_length": len(executive_summary) + sum(len(s.content) for s in sections)
            }
        )
    
    async def _generate_executive_summary(self, target, orchestrated_results, 
                                        categorized_insights) -> str:
        """Generate executive summary"""
        # Extract key findings
        top_insights = categorized_insights.get("top_insights", [])
        key_findings = []
        for insight in top_insights[:5]:
            key_findings.append(f"• {insight['insight']}")
        
        # Determine overall risk level
        overall_risk = orchestrated_results.risk_assessment.get("overall_risk_score", 0)
        if overall_risk > 7:
            risk_level = "ALTO"
            risk_summary = "Foram identificados riscos significativos que requerem atenção imediata."
        elif overall_risk > 4:
            risk_level = "MÉDIO"
            risk_summary = "Riscos moderados identificados, monitoramento recomendado."
        else:
            risk_level = "BAIXO"
            risk_summary = "Perfil de risco baixo, sem preocupações imediatas."
        
        # Top recommendations
        recommendations = categorized_insights.get("recommendations", [])
        top_recommendations = []
        for rec in recommendations[:3]:
            top_recommendations.append(f"• {rec['recommendation']}")
        
        # Calculate overall confidence
        consensus_metrics = orchestrated_results.consensus_metrics
        overall_confidence = consensus_metrics.get("overall_consensus", 0.5)
        
        return self.report_templates["executive_summary"].format(
            target_name=target.name,
            target_email=target.email,
            report_date=datetime.now().strftime("%d/%m/%Y %H:%M"),
            overall_confidence=overall_confidence,
            key_findings="\n".join(key_findings) if key_findings else "Nenhuma descoberta significativa.",
            risk_level=risk_level,
            risk_summary=risk_summary,
            top_recommendations="\n".join(top_recommendations) if top_recommendations else "Nenhuma recomendação específica."
        )
    
    async def _generate_personality_section(self, personality_data: Dict[str, Any]) -> ReportSection:
        """Generate personality analysis section"""
        if not personality_data:
            return ReportSection(
                title="Análise de Personalidade",
                content="Dados de personalidade não disponíveis.",
                subsections=[],
                metadata={}
            )
        
        # Format personality traits
        traits = personality_data.get("consolidated_traits", {})
        trait_lines = []
        for trait, score in traits.items():
            if isinstance(score, (int, float)):
                trait_lines.append(f"• **{trait.title()}:** {score:.2f} ({self._interpret_trait_score(score)})")
        
        # Format communication style
        comm_style = personality_data.get("communication_style", {})
        style_lines = []
        for aspect, value in comm_style.items():
            style_lines.append(f"• **{aspect.title()}:** {value}")
        
        # Format behavioral indicators
        indicators = personality_data.get("behavioral_indicators", [])
        indicator_lines = [f"• {indicator}" for indicator in indicators[:5]]
        
        content = self.report_templates["personality_section"].format(
            personality_traits="\n".join(trait_lines) if trait_lines else "Não disponível",
            communication_style="\n".join(style_lines) if style_lines else "Não disponível",
            behavioral_indicators="\n".join(indicator_lines) if indicator_lines else "Não disponível",
            confidence=personality_data.get("trait_confidence", 0.5),
            sources_count=personality_data.get("sources_count", 0)
        )
        
        return ReportSection(
            title="Análise de Personalidade",
            content=content,
            subsections=[],
            metadata=personality_data
        )
    
    async def _generate_behavioral_section(self, behavioral_data: Dict[str, Any]) -> ReportSection:
        """Generate behavioral predictions section"""
        if not behavioral_data:
            return ReportSection(
                title="Predições Comportamentais",
                content="Dados comportamentais não disponíveis.",
                subsections=[],
                metadata={}
            )
        
        # Format decision patterns
        decision_making = behavioral_data.get("decision_making", {})
        decision_lines = []
        for aspect, pattern in decision_making.items():
            decision_lines.append(f"• **{aspect.replace('_', ' ').title()}:** {pattern}")
        
        # Format work patterns
        work_patterns = behavioral_data.get("work_patterns", {})
        work_lines = []
        for pattern, value in work_patterns.items():
            work_lines.append(f"• **{pattern.replace('_', ' ').title()}:** {value}")
        
        # Format adaptation capacity
        adaptation = behavioral_data.get("adaptation_capacity", {})
        adaptation_lines = []
        for metric, score in adaptation.items():
            if isinstance(score, (int, float)):
                adaptation_lines.append(f"• **{metric.replace('_', ' ').title()}:** {score:.2f}")
        
        # Format interaction patterns
        interaction = behavioral_data.get("interaction_patterns", {})
        interaction_lines = []
        for pattern, value in interaction.items():
            interaction_lines.append(f"• **{pattern.replace('_', ' ').title()}:** {value}")
        
        content = self.report_templates["behavioral_section"].format(
            decision_patterns="\n".join(decision_lines) if decision_lines else "Não disponível",
            work_patterns="\n".join(work_lines) if work_lines else "Não disponível",
            adaptation_capacity="\n".join(adaptation_lines) if adaptation_lines else "Não disponível",
            interaction_patterns="\n".join(interaction_lines) if interaction_lines else "Não disponível",
            confidence=behavioral_data.get("prediction_confidence", 0.5)
        )
        
        return ReportSection(
            title="Predições Comportamentais",
            content=content,
            subsections=[],
            metadata=behavioral_data
        )
    
    async def _generate_risk_section(self, risk_data: Dict[str, Any]) -> ReportSection:
        """Generate risk assessment section"""
        if not risk_data:
            return ReportSection(
                title="Avaliação de Riscos",
                content="Dados de risco não disponíveis.",
                subsections=[],
                metadata={}
            )
        
        overall_risk = risk_data.get("overall_risk_score", 0)
        
        # Format risk categories
        risk_categories = risk_data.get("risk_categories", {})
        category_lines = []
        for category, score in risk_categories.items():
            if isinstance(score, (int, float)):
                risk_level = self._interpret_risk_score(score)
                category_lines.append(f"• **{category.replace('_', ' ').title()}:** {score:.1f}/10 ({risk_level})")
        
        # Format top risk factors
        risk_factors = risk_data.get("top_risk_factors", [])
        factor_lines = []
        for factor in risk_factors[:5]:
            if isinstance(factor, dict):
                factor_name = factor.get("factor", "Fator desconhecido")
                impact = factor.get("impact", "médio")
                factor_lines.append(f"• **{factor_name}** - Impacto: {impact}")
        
        # Format recommendations
        recommendations = risk_data.get("recommendations", [])
        rec_lines = [f"• {rec}" for rec in recommendations[:5]]
        
        content = self.report_templates["risk_section"].format(
            overall_risk_score=overall_risk,
            risk_categories="\n".join(category_lines) if category_lines else "Não disponível",
            top_risk_factors="\n".join(factor_lines) if factor_lines else "Nenhum fator de risco específico identificado",
            risk_recommendations="\n".join(rec_lines) if rec_lines else "Nenhuma recomendação específica",
            confidence=risk_data.get("risk_confidence", 0.5)
        )
        
        return ReportSection(
            title="Avaliação de Riscos",
            content=content,
            subsections=[],
            metadata=risk_data
        )
    
    async def _generate_timeline_section(self, timeline) -> ReportSection:
        """Generate timeline analysis section"""
        if not timeline or not hasattr(timeline, 'timeline_analysis'):
            return ReportSection(
                title="Análise de Timeline",
                content="Dados de timeline não disponíveis.",
                subsections=[],
                metadata={}
            )
        
        timeline_analysis = timeline.timeline_analysis
        
        # Extract timeline data
        timeline_span = timeline_analysis.get("timeline_span_years", 0)
        total_events = timeline_analysis.get("total_events", 0)
        event_density = timeline_analysis.get("event_density_per_year", 0)
        
        # Format key milestones
        if hasattr(timeline, 'events'):
            key_events = sorted(timeline.events, key=lambda x: x.importance_score if hasattr(x, 'importance_score') else 0, reverse=True)[:5]
            milestone_lines = []
            for event in key_events:
                milestone_lines.append(f"• **{event.date}:** {event.description}")
        else:
            milestone_lines = ["Marcos não disponíveis"]
        
        # Format peak periods
        peak_periods = timeline_analysis.get("peak_activity_periods", [])
        peak_lines = []
        for period in peak_periods[:3]:
            year = period.get("year", "Desconhecido")
            count = period.get("event_count", 0)
            peak_lines.append(f"• **{year}:** {count} eventos")
        
        # Format patterns
        patterns = []
        if hasattr(timeline, 'patterns_identified'):
            pattern_data = timeline.patterns_identified
            for pattern_type, pattern_info in pattern_data.items():
                if isinstance(pattern_info, dict):
                    patterns.append(f"• **{pattern_type.replace('_', ' ').title()}:** Identificado")
        
        content = self.report_templates["timeline_section"].format(
            timeline_span=timeline_span,
            total_events=total_events,
            event_density=event_density,
            key_milestones="\n".join(milestone_lines),
            peak_periods="\n".join(peak_lines) if peak_lines else "Nenhum período de pico identificado",
            timeline_patterns="\n".join(patterns) if patterns else "Nenhum padrão específico identificado",
            confidence=getattr(timeline.validation_result, 'confidence_score', 0.5) if hasattr(timeline, 'validation_result') else 0.5
        )
        
        return ReportSection(
            title="Análise de Timeline",
            content=content,
            subsections=[],
            metadata=timeline_analysis
        )
    
    async def _generate_insights_section(self, categorized_insights: Dict[str, Any]) -> ReportSection:
        """Generate insights section"""
        if not categorized_insights:
            return ReportSection(
                title="Insights Categorizados",
                content="Insights não disponíveis.",
                subsections=[],
                metadata={}
            )
        
        # Format top insights
        top_insights = categorized_insights.get("top_insights", [])
        insight_lines = []
        for i, insight in enumerate(top_insights, 1):
            category = insight.get("category", "").replace("_", " ").title()
            insight_text = insight.get("insight", "")
            confidence = insight.get("confidence", 0)
            insight_lines.append(f"{i}. **[{category}]** {insight_text} (Confiança: {confidence:.1%})")
        
        # Format categorized insights
        categorized_data = categorized_insights.get("categorized_insights", {})
        category_lines = []
        for category_name, category_info in categorized_data.items():
            if hasattr(category_info, 'total_insights'):
                total = category_info.total_insights
                high_conf = category_info.high_confidence_count
                category_display = category_name.replace("_", " ").title()
                category_lines.append(f"• **{category_display}:** {total} insights ({high_conf} alta confiança)")
        
        # Format recommendations
        recommendations = categorized_insights.get("recommendations", [])
        rec_lines = []
        for i, rec in enumerate(recommendations[:5], 1):
            rec_text = rec.get("recommendation", "")
            priority = rec.get("priority", "medium")
            rec_lines.append(f"{i}. **[{priority.upper()}]** {rec_text}")
        
        content = self.report_templates["insights_section"].format(
            top_insights="\n".join(insight_lines) if insight_lines else "Nenhum insight disponível",
            categorized_insights="\n".join(category_lines) if category_lines else "Nenhuma categorização disponível",
            actionable_recommendations="\n".join(rec_lines) if rec_lines else "Nenhuma recomendação disponível"
        )
        
        return ReportSection(
            title="Insights Categorizados",
            content=content,
            subsections=[],
            metadata=categorized_insights
        )
    
    async def _generate_technical_section(self, orchestrated_results, timeline, 
                                        categorized_insights) -> ReportSection:
        """Generate technical details section"""
        consensus_metrics = orchestrated_results.consensus_metrics
        
        # Data sources
        data_sources = ["Redes Sociais", "Registros Públicos", "Redes Profissionais", "Pegada Digital"]
        
        # AI services
        ai_services = ["OpenAI GPT", "Anthropic Claude", "Google AI"]
        
        # Conflicts resolved
        conflicts = orchestrated_results.conflicts_resolved
        conflict_lines = []
        for conflict in conflicts[:3]:
            if hasattr(conflict, 'field_name'):
                conflict_lines.append(f"• Campo: {conflict.field_name} - Método: {conflict.resolution_method}")
        
        # Data quality metrics
        quality_lines = []
        if hasattr(timeline, 'validation_result'):
            validation = timeline.validation_result
            quality_lines.append(f"• Pontuação de Validação da Timeline: {validation.confidence_score:.1%}")
            if validation.inconsistencies:
                quality_lines.append(f"• Inconsistências Detectadas: {len(validation.inconsistencies)}")
        
        metadata = categorized_insights.get("metadata", {})
        total_insights = metadata.get("total_insights", 0)
        if total_insights > 0:
            quality_lines.append(f"• Total de Insights Gerados: {total_insights}")
        
        content = self.report_templates["technical_section"].format(
            data_sources="\n".join([f"• {source}" for source in data_sources]),
            ai_services="\n".join([f"• {service}" for service in ai_services]),
            success_rate=consensus_metrics.get("success_rate", 0),
            average_confidence=consensus_metrics.get("average_confidence", 0),
            service_agreement=consensus_metrics.get("service_agreement", 0),
            overall_consensus=consensus_metrics.get("overall_consensus", 0),
            conflicts_resolved="\n".join(conflict_lines) if conflict_lines else "Nenhum conflito significativo",
            data_quality_metrics="\n".join(quality_lines) if quality_lines else "Métricas não disponíveis"
        )
        
        return ReportSection(
            title="Detalhes Técnicos",
            content=content,
            subsections=[],
            metadata={
                "consensus_metrics": consensus_metrics,
                "conflicts_count": len(conflicts)
            }
        )
    
    async def _save_report_to_file(self, target, executive_summary: str, 
                                 sections: List[ReportSection], title: str) -> str:
        """Save report to file"""
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in target.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"relatorio_{safe_name}_{timestamp}.md"
        file_path = os.path.join(self.output_dir, filename)
        
        # Combine all content
        full_content = executive_summary + "\n\n"
        for section in sections:
            full_content += section.content + "\n\n"
        
        # Add footer
        full_content += f"""
---
*Relatório gerado automaticamente em {datetime.now().strftime("%d/%m/%Y às %H:%M:%S")}*
*Sistema de Investigação Automatizada com IA*
        """
        
        # Save to file
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(full_content)
            logger.info(f"Report saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
            return ""
    
    # Helper methods
    def _interpret_trait_score(self, score: float) -> str:
        """Interpret personality trait score"""
        if score > 0.7:
            return "Alto"
        elif score > 0.4:
            return "Médio"
        else:
            return "Baixo"
    
    def _interpret_risk_score(self, score: float) -> str:
        """Interpret risk score"""
        if score > 7:
            return "Alto"
        elif score > 4:
            return "Médio"
        else:
            return "Baixo"
    
    async def generate_json_report(self, target, orchestrated_results, timeline, 
                                 categorized_insights) -> str:
        """Generate JSON format report for programmatic use"""
        report_data = {
            "target": {
                "name": target.name,
                "email": target.email,
                "additional_data": target.additional_data
            },
            "generation_timestamp": datetime.now().isoformat(),
            "personality_profile": orchestrated_results.personality_profile,
            "behavioral_predictions": orchestrated_results.behavioral_predictions,
            "risk_assessment": orchestrated_results.risk_assessment,
            "timeline_analysis": timeline.timeline_analysis if timeline else {},
            "categorized_insights": categorized_insights,
            "consensus_metrics": orchestrated_results.consensus_metrics,
            "metadata": {
                "report_version": "1.0",
                "ai_services_used": ["OpenAI", "Anthropic", "Google AI"],
                "data_sources": ["social_media", "public_records", "professional_networks"]
            }
        }
        
        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in target.name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"relatorio_{safe_name}_{timestamp}.json"
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"JSON report saved to: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save JSON report: {e}")
            return ""
