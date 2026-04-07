import yaml
import json
import pandas as pd
from ..causal_analysis.correlation import analyze_data
from ..agents.human_proxy import HumanProxy
from ..agents.causal_oracle import CausalOracle
from ..agents.hypothesizer import Hypothesizer
from ..agents.confounder_miner import ConfounderMiner
from ..critics.exclusion_critic import ExclusionCritic
from ..critics.independence_critic import IndependenceCritic
from ..agents.grounder import Grounder
from ..analysis.iv_estimator import IVEstimator
from ..llm.client import LLMClient
import os

class IVCoScientist:
    def __init__(self, config):
        # Accept either config dict or config path
        if isinstance(config, str):
            with open(config, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
        
        self.dataset_path = self.config['dataset']['path']
        self.llm_client = LLMClient(self.config['llm'])
        
        # Initialize agents
        self.human_proxy = HumanProxy(self.llm_client, max_pairs=self.config['agents']['human_proxy']['max_pairs'])
        self.causal_oracle = CausalOracle(self.llm_client, self.dataset_path)
        self.hypothesizer = Hypothesizer(self.llm_client, k=self.config['agents']['hypothesizer']['k_ivs'])
        self.confounder_miner = ConfounderMiner(self.llm_client, j=self.config['agents']['confounder_miner']['j_confounders'])
        self.exclusion_critic = ExclusionCritic(self.llm_client)
        self.independence_critic = IndependenceCritic(self.llm_client)
        self.grounder = Grounder(self.dataset_path, threshold=self.config['agents']['grounder']['threshold'])
        self.estimator = IVEstimator(self.dataset_path, f_threshold=self.config['estimation']['f_threshold'])
    
    def run_discovery(self):
        """Run full IV discovery pipeline"""
        print("🔍 Starting IV Co-Scientist Discovery...")
        
        # Stage 1: PreSelector - Filter correlations
        print("\n📊 Stage 1: PreSelector - Analyzing correlations...")
        correlation_pairs = self._preselector()
        print(f"Found {len(correlation_pairs)} correlated pairs")
        
        if not correlation_pairs:
            return {"error": "No significant correlations found"}
        
        # Stage 2: HumanProxy - Select meaningful pairs
        print("\n🧠 Stage 2: HumanProxy - Selecting meaningful pairs...")
        meaningful_pairs = self.human_proxy.select_meaningful_pairs(correlation_pairs)
        print(f"Selected {len(meaningful_pairs)} meaningful pairs")
        
        # Stage 3: CausalOracle - Infer direction
        print("\n🔮 Stage 3: CausalOracle - Inferring causal direction...")
        causal_pairs = []
        for pair in meaningful_pairs:
            direction = self.causal_oracle.infer_direction(pair['variable1'], pair['variable2'])
            if direction:
                causal_pairs.append({
                    'treatment': direction[0],
                    'outcome': direction[1],
                    'original_pair': pair
                })
        print(f"Identified {len(causal_pairs)} directional pairs")
        
        # Process each causal pair
        results = []
        for i, causal_pair in enumerate(causal_pairs):
            print(f"\n🎯 Processing pair {i+1}/{len(causal_pairs)}: {causal_pair['treatment']} → {causal_pair['outcome']}")
            result = self._process_causal_pair(causal_pair)
            if result:
                results.append(result)
        
        return {
            'total_correlations': len(correlation_pairs),
            'meaningful_pairs': len(meaningful_pairs),
            'causal_pairs': len(causal_pairs),
            'successful_discoveries': len(results),
            'results': results
        }
    
    def _preselector(self):
        """Filter variable pairs by correlation and sample size"""
        # Check if correlation results already exist
        correlation_file = "gapminder_correlations.csv"
        
        if not os.path.exists(correlation_file):
            print("Running correlation analysis first...")
            from ..analyze_gapminder import analyze_gapminder_folder
            analyze_gapminder_folder(self.dataset_path)
        
        # Load existing correlation results
        import pandas as pd
        df = pd.read_csv(correlation_file, sep=';')
        
        # Convert to expected format and filter
        correlation_pairs = []
        for _, row in df.iterrows():
            if self._passes_filter(row.to_dict()):
                correlation_pairs.append(row.to_dict())
        
        return correlation_pairs[:self.config['preselector'].get('max_pairs', 20)]
    
    def _passes_filter(self, result):
        """Check if correlation pair passes thresholds"""
        return (abs(result['correlation']) > self.config['preselector']['correlation_threshold'] and
                result['data_points'] > self.config['preselector']['min_data_points'])
    
    def _process_causal_pair(self, causal_pair):
        """Process single causal pair through full pipeline"""
        treatment = causal_pair['treatment']
        outcome = causal_pair['outcome']
        
        # Stage 4: Generate IVs and confounders
        print(f"  📝 Generating IVs for {treatment} → {outcome}")
        proposed_ivs = self.hypothesizer.propose_ivs(treatment, outcome)
        confounders = self.confounder_miner.identify_confounders(treatment, outcome)
        
        # Stage 5: Validate IVs
        print(f"  ✅ Validating {len(proposed_ivs)} proposed IVs")
        valid_ivs = self._validate_ivs(proposed_ivs, treatment, outcome, confounders)
        
        if not valid_ivs:
            print(f"  ❌ No valid IVs found")
            return None
        
        # Stage 6: Ground IVs to dataset
        print(f"  🎯 Grounding {len(valid_ivs)} valid IVs to dataset")
        grounded_ivs = self.grounder.ground_ivs(valid_ivs)
        
        if not grounded_ivs:
            print(f"  ❌ No IVs could be grounded to dataset")
            return None
        
        # Stage 7: Estimate causal effects
        print(f"  📈 Estimating effects with {len(grounded_ivs)} grounded IVs")
        instrument_vars = [pair[1] for pair in grounded_ivs]  # Extract proxy variables
        estimation_results = self.estimator.estimate_iv_effect(treatment, outcome, instrument_vars)
        
        valid_estimates = [est for est in estimation_results if est and est['relevance_check']]
        
        return {
            'treatment': treatment,
            'outcome': outcome,
            'proposed_ivs': proposed_ivs,
            'valid_ivs': valid_ivs,
            'grounded_ivs': grounded_ivs,
            'estimation_results': valid_estimates,
            'success': len(valid_estimates) > 0
        }
    
    def _validate_ivs(self, proposed_ivs, treatment, outcome, confounders):
        """Validate IVs using critics (parallel evaluation)"""
        exclusion_results = {}
        independence_results = {}
        
        # Exclusion critic
        for iv in proposed_ivs:
            exclusion_results[iv] = self.exclusion_critic.validate_exclusion(iv, treatment, outcome, confounders)
        
        # Independence critic  
        for iv in proposed_ivs:
            independence_results[iv] = self.independence_critic.validate_independence(iv, treatment, outcome, confounders)
        
        # Keep only IVs that pass both tests
        valid_ivs = []
        for iv in proposed_ivs:
            if exclusion_results[iv] and independence_results[iv]:
                valid_ivs.append(iv)
        
        return valid_ivs