import pandas as pd
import numpy as np
import pickle

class FraudPreprocessor:
    def __init__(self, stats_path='output_v2/feature_stats.pkl', feature_names_path='output_v2/feature_names.pkl'):
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
            
    def group_email(self, email):
        if pd.isna(email): return 'nan'
        email = str(email).lower()
        if 'gmail' in email: return 'google'
        if 'yahoo' in email: return 'yahoo'
        if 'hotmail' in email or 'outlook' in email or 'msn' in email: return 'microsoft'
        if 'icloud' in email or 'me.com' in email or 'mac.com' in email: return 'apple'
        return 'other'

    def transform(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        df = raw_data.copy()
        
        # 1. Time Features
        if 'TransactionDT' in df.columns:
            df['Transaction_hour'] = np.floor(df['TransactionDT'] / 3600) % 24
            df['Transaction_day'] = np.floor(df['TransactionDT'] / (3600 * 24))
            df['Transaction_dow'] = (df['Transaction_day'] % 7)
            
        # 2. Aggregates (Lookups)
        for col in ['card1', 'card2', 'addr1']:
            if col in df.columns:
                val = df[col].iloc[0]
                stat = self.stats[col].get(val, {'mean': 0, 'std': 0})
                df[f'{col}_Amt_mean'] = stat['mean']
                df[f'{col}_Amt_std'] = stat['std']
                df[f'{col}_Amt_diff'] = df['TransactionAmt'] - stat['mean']
            else:
                df[f'{col}_Amt_mean'] = 0
                df[f'{col}_Amt_std'] = 0
                df[f'{col}_Amt_diff'] = 0

        # 3. Frequency
        if 'card1' in df.columns:
            val = df['card1'].iloc[0]
            # Approximate count from stats if not available
            df['card1_count'] = len(self.stats['card1'].get(val, {})) 
        else:
            df['card1_count'] = 0

        # 4. Email Bins
        if 'P_emaildomain' in df.columns:
            df['P_emaildomain_bin'] = df['P_emaildomain'].apply(self.group_email)
        else:
            df['P_emaildomain_bin'] = 'nan'

        # 5. Categorical Encoding (Simple codes)
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].astype('category').cat.codes

        # 6. Alignment
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = -999
        
        df = df[self.feature_names]
        df = df.fillna(-999)
        
        return df
