
def format_pipeline_config_for_prompt(cfg):
    return f"""
Model Type: {cfg['model_type']}
Target Column: {cfg['target_column']}
Random Seed: {cfg['random_seed']}
Split: {int((1 - cfg['test_size'] - cfg.get('val_size', 0)) * 100)}% Train / {int(cfg.get('val_size', 0) * 100)}% Val / {int(cfg['test_size'] * 100)}% Test
Evaluation Metrics: {", ".join(cfg['evaluation_metrics'])}
"""