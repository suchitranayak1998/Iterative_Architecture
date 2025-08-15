# core/execution.py
"""
Code execution engines - same as original for compatibility
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import contextlib
import base64
import re
import psutil
import warnings
import os
import traceback
import textwrap
warnings.filterwarnings('ignore')

class DeveloperExecutor:
    def __init__(self, df):
        self.df = df

    def get_column_catalog(self):
        info = []
        df = self.df.copy()
        column_info = df.dtypes.to_string()
        summary_stats = df.describe(include='all').to_string()
        col_names = ", ".join(df.columns)
        index_cols = list(df.index.names) if df.index.name or df.index.names else []  
        index_info = f"Index columns: {index_cols}" if index_cols else "No index column set."

        catalog = f"""## Dataset

        ### Schema:
        {column_info}

        ### Summary Statistics:
        {summary_stats}

        ### Column Names:
        {col_names}

        ### Index Info:
        {index_info}"""
        
        return catalog
    
    def extract_code(self, text: str) -> str:
        match = re.search(r"```python(.*?)```", text, re.DOTALL)
        return match.group(1).strip() if match else ""

    def run_code(self, code: str):
        local_env = {'df': self.df, 'pd': pd, 'plt': plt, 'os': os, 'sns': sns, 'np': np}

        output_buffer = io.StringIO()
        images = []

        try:
            with contextlib.redirect_stdout(output_buffer):
                wrapped_code = textwrap.dedent(code)
                exec(wrapped_code, local_env, local_env)

                if "df" in local_env:
                    #check if the syntax is correct
                    self.df = local_env["df"]

                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    buf = io.BytesIO()
                    plt.figure(fig_num).savefig(buf, format='png', bbox_inches='tight')
                    buf.seek(0)
                    image_base64 = base64.b64encode(buf.read()).decode("utf-8")
                    images.append(image_base64)
                    plt.close(fig_num)
                
                return True, output_buffer.getvalue().strip(), images
            
        except Exception as e:
            error_trace = traceback.format_exc()
            print(error_trace)
            return False, error_trace, []