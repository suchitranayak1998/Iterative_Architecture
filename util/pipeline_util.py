# util/pipeline_util.py
"""
Pipeline utility functions for iterative multi-agent system
"""

from typing import Dict, Any, List
import json

def format_pipeline_config_for_prompt(cfg: Dict[str, Any]) -> str:
    """Format pipeline configuration for inclusion in prompts"""
    return f"""
Architecture: {cfg.get('architecture', '3-agent iterative')}
Model Type: {cfg['model_type']}
Target Column: {cfg['target_column']}
Random Seed: {cfg['random_seed']}
Split: {int((1 - cfg['test_size'] - cfg.get('val_size', 0)) * 100)}% Train / {int(cfg.get('val_size', 0) * 100)}% Val / {int(cfg['test_size'] * 100)}% Test
Evaluation Metrics: {", ".join(cfg['evaluation_metrics'])}
Process Steps: {cfg.get('process_steps', 4)}
Max Debug Retries: {cfg.get('max_debug_retries', 3)}
"""

def validate_iterative_config(cfg: Dict[str, Any]) -> List[str]:
    """Validate iterative pipeline configuration"""
    errors = []
    
    required_fields = [
        'target_column', 'model_type', 'evaluation_metrics', 
        'random_seed', 'test_size'
    ]
    
    for field in required_fields:
        if field not in cfg:
            errors.append(f"Missing required field: {field}")
    
    # Iterative-specific validation
    if 'architecture' in cfg and cfg['architecture'] != '3-agent iterative':
        errors.append(f"Expected '3-agent iterative' architecture, got: {cfg['architecture']}")
    
    if 'agent_roles' in cfg:
        expected_roles = ['Planner', 'Developer', 'Auditor']
        if cfg['agent_roles'] != expected_roles:
            errors.append(f"Expected agent roles {expected_roles}, got: {cfg['agent_roles']}")
    
    if 'process_steps' in cfg and cfg['process_steps'] != 4:
        errors.append(f"Expected 4 process steps for iterative workflow, got: {cfg['process_steps']}")
    
    return errors

def extract_agent_metrics(agent_interaction_log: List[Dict]) -> Dict[str, Any]:
    """Extract metrics from agent interaction logs"""
    if not agent_interaction_log:
        return {}
    
    # Count interactions by agent and step
    agent_counts = {}
    step_counts = {}
    
    for interaction in agent_interaction_log:
        agent = interaction.get('agent_name', 'Unknown')
        step = interaction.get('step', 'Unknown')
        
        agent_counts[agent] = agent_counts.get(agent, 0) + 1
        step_counts[step] = step_counts.get(step, 0) + 1
    
    return {
        "total_interactions": len(agent_interaction_log),
        "agent_interaction_counts": agent_counts,
        "step_interaction_counts": step_counts,
        "unique_agents": len(agent_counts),
        "unique_steps": len(step_counts)
    }

def format_iterative_summary(process_data: Dict[str, Any]) -> str:
    """Format iterative process data for reporting"""
    if not process_data:
        return "No iterative process data available"
    
    summary = "## Iterative Process Summary\n\n"
    
    # Step 1: Planner
    if 'planner_output' in process_data:
        planner = process_data['planner_output']
        summary += f"**Step 1 - {planner.get('agent', 'Planner')}:** Strategic planning completed\n"
    
    # Step 2: Initial Developer
    if 'initial_developer_output' in process_data:
        initial_dev = process_data['initial_developer_output']
        summary += f"**Step 2 - {initial_dev.get('agent', 'Developer')}:** Initial implementation completed\n"
    
    # Step 3: Auditor
    if 'auditor_feedback' in process_data:
        auditor = process_data['auditor_feedback']
        summary += f"**Step 3 - {auditor.get('agent', 'Auditor')}:** Quality review and feedback provided\n"
    
    # Step 4: Final Developer
    if 'final_developer_output' in process_data:
        final_dev = process_data['final_developer_output']
        summary += f"**Step 4 - {final_dev.get('agent', 'Developer')}:** Refined implementation completed\n"
    
    summary += f"\n**Process Status:** {'✅ Complete' if process_data.get('process_complete') else '❌ Incomplete'}\n"
    
    return summary

def calculate_process_quality_score(results: List[Dict]) -> Dict[str, float]:
    """Calculate quality scores for iterative processes"""
    if not results:
        return {"overall_score": 0.0}
    
    scores = {
        "planning_quality": 0.0,
        "implementation_quality": 0.0, 
        "audit_quality": 0.0,
        "refinement_quality": 0.0,
        "execution_success_rate": 0.0,
        "overall_score": 0.0
    }
    
    total_processes = len(results)
    successful_executions = 0
    
    for result in results:
        if result.get("success", False):
            successful_executions += 1
        
        # Check for iterative process completeness
        if "iterative_process" in result:
            iterative_data = result["iterative_process"]
            
            # Planning quality (presence and completeness)
            if iterative_data.get("planner_output", {}).get("planning_instructions"):
                scores["planning_quality"] += 1
            
            # Implementation quality (code generation)
            if iterative_data.get("final_developer_output", {}).get("final_implementation"):
                scores["implementation_quality"] += 1
            
            # Audit quality (feedback provided)
            if iterative_data.get("auditor_feedback", {}).get("audit_feedback"):
                scores["audit_quality"] += 1
            
            # Refinement quality (improvement from initial to final)
            initial_impl = iterative_data.get("initial_developer_output", {}).get("implementation", "")
            final_impl = iterative_data.get("final_developer_output", {}).get("final_implementation", "")
            if final_impl and len(final_impl) > len(initial_impl):
                scores["refinement_quality"] += 1
    
    # Calculate success rate
    scores["execution_success_rate"] = successful_executions / total_processes if total_processes > 0 else 0
    
    # Normalize other scores
    for key in ["planning_quality", "implementation_quality", "audit_quality", "refinement_quality"]:
        scores[key] = scores[key] / total_processes if total_processes > 0 else 0
    
    # Calculate overall score
    scores["overall_score"] = sum(scores.values()) / len(scores)
    
    return scores

def format_agent_performance_report(pipeline_state) -> str:
    """Generate agent performance report from pipeline state"""
    if not hasattr(pipeline_state, 'agent_interaction_log'):
        return "No agent interaction data available"
    
    metrics = extract_agent_metrics(pipeline_state.agent_interaction_log)
    
    if not metrics:
        return "No agent metrics to report"
    
    report = "## Agent Performance Report\n\n"
    report += f"**Total Interactions:** {metrics['total_interactions']}\n"
    report += f"**Active Agents:** {metrics['unique_agents']}\n"
    report += f"**Process Steps:** {metrics['unique_steps']}\n\n"
    
    report += "### Agent Interaction Counts\n"
    for agent, count in metrics['agent_interaction_counts'].items():
        report += f"- {agent}: {count} interactions\n"
    
    report += "\n### Step Distribution\n"
    for step, count in metrics['step_interaction_counts'].items():
        report += f"- {step}: {count} interactions\n"
    
    return report

class IterativePipelineUtils:
    """Utility class for iterative pipeline operations"""
    
    @staticmethod
    def validate_agent_sequence(agent_sequence: List[str]) -> bool:
        """Validate that agent sequence follows iterative pattern"""
        expected_pattern = ['Planner', 'Developer', 'Auditor', 'Developer']
        
        if len(agent_sequence) < 3:
            return False
        
        # Check for required roles
        has_planner = any('planner' in agent.lower() for agent in agent_sequence)
        has_developer = any('developer' in agent.lower() for agent in agent_sequence)
        has_auditor = any('auditor' in agent.lower() for agent in agent_sequence)
        
        return has_planner and has_developer and has_auditor
    
    @staticmethod
    def extract_improvement_metrics(initial_output: str, final_output: str) -> Dict[str, Any]:
        """Extract metrics showing improvement from initial to final output"""
        return {
            "initial_length": len(initial_output) if initial_output else 0,
            "final_length": len(final_output) if final_output else 0,
            "length_improvement": len(final_output) - len(initial_output) if both_exist(initial_output, final_output) else 0,
            "improvement_ratio": len(final_output) / len(initial_output) if initial_output and final_output and len(initial_output) > 0 else 1.0
        }

def both_exist(initial: str, final: str) -> bool:
    """Helper function to check if both initial and final outputs exist"""
    return bool(initial and final)