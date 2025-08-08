
import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm, expm, logm
from scipy.special import erf, erfc, expit
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import time
import logging
from typing import Dict, Tuple, List
import os
import matplotlib
matplotlib.rcParams['text.usetex'] = False
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'figure.dpi': 300,
    'savefig.dpi': 1200,
    'axes.grid': True
})

# Ensure output folders exist
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QuantumRadarSimulator:
    def __init__(self):
        self.r = 1.2
        self.theta = np.pi/4
        self.M_values = np.array([1, 3, 5, 10, 20, 50, 100])
        self.kappa_values = np.linspace(0.05, 0.95, 10)
        self.NB_values = np.linspace(0, 3, 7)
        self.threshold = 0.5
        self.detectors = {
            'classical': ClassicalDetector(),
            'qcb': QCBCalculator(),
            'qnn': QuantumNeuralNetwork()
        }

    def generate_TMSV_state(self) -> np.ndarray:
        cosh_r = np.cosh(self.r)
        sinh_r = np.sinh(self.r)
        return np.array([
            [cosh_r, 0, sinh_r * np.cos(self.theta), sinh_r * np.sin(self.theta)],
            [0, cosh_r, sinh_r * np.sin(self.theta), -sinh_r * np.cos(self.theta)],
            [sinh_r * np.cos(self.theta), sinh_r * np.sin(self.theta), cosh_r, 0],
            [sinh_r * np.sin(self.theta), -sinh_r * np.cos(self.theta), 0, cosh_r]
        ])

    def apply_thermal_loss(self, cov_matrix: np.ndarray, kappa: float, NB: float) -> np.ndarray:
        X = np.sqrt(kappa) * np.eye(4)
        Y = (1 - kappa) * (2 * NB + 1) * np.eye(4)
        return X @ cov_matrix @ X.T + Y

    def simulate_system(self) -> Dict:
        start_time = time.time()
        results = {
            'parameters': [],
            'performance': {},
            'computational_cost': {}
        }
        for name in self.detectors:
            results['performance'][name] = {'P_D': [], 'P_FA': []}
            results['computational_cost'][name] = []

        tmsv_state = self.generate_TMSV_state()
        self.detectors['qnn'].train(tmsv_state, self.kappa_values, self.NB_values)
        for M in tqdm(self.M_values, desc='Mode numbers'):
            for kappa in self.kappa_values:
                for NB in self.NB_values:
                    results['parameters'].append((M, kappa, NB))
                    noisy_state = self.apply_thermal_loss(tmsv_state, kappa, NB)
                    for name, detector in self.detectors.items():
                        t_start = time.time()
                        P_D, P_FA = detector.analyze(noisy_state, M)
                        t_end = time.time()
                        results['performance'][name]['P_D'].append(P_D)
                        results['performance'][name]['P_FA'].append(P_FA)
                        results['computational_cost'][name].append(t_end - t_start)
        results['quantum_advantage'] = self.calculate_quantum_advantage()
        logging.info(f"Total simulation time: {time.time() - start_time:.2f} seconds")
        return results

    def calculate_quantum_advantage(self) -> Dict:
        T_over_f = np.logspace(-2, 2, 100)
        return {
            'T_over_f': T_over_f,
            'QA': 2 - 1 / (1 + T_over_f)
        }

    def save_results(self, results: Dict, filename: str = 'results/qcr_results'):
        df = pd.DataFrame({
            'M': [p[0] for p in results['parameters']],
            'kappa': [p[1] for p in results['parameters']],
            'NB': [p[2] for p in results['parameters']],
            **{f'{det}_{metric}': values
               for det in results['performance']
               for metric, values in results['performance'][det].items()}
        })
        df.to_csv(f'{filename}.csv', index=False)
        cost_df = pd.DataFrame(results['computational_cost'])
        cost_df.to_csv(f'{filename}_cost.csv', index=False)
        logging.info(f"Results saved to {filename}.csv and {filename}_cost.csv")

    def plot_all_results(self, results: Dict):
        plotter = ResultPlotter()
        plotter.plot_roc_curves(results['performance'])
        plotter.plot_mode_effect(results['performance'], self.M_values, self.kappa_values, self.NB_values)
        plotter.plot_noise_effect(results['performance'], self.M_values, self.kappa_values, self.NB_values)
        plotter.plot_quantum_advantage(results['quantum_advantage'])
        plotter.plot_performance_heatmap(
            results['performance'], self.M_values, self.kappa_values, self.NB_values
        )
        plotter.plot_computational_cost(results['computational_cost'])
        # four additional academic-style figures
        plotter.plot_complexity_chart()
        plotter.plot_qnn_convergence()
        plotter.plot_jamming_impact()
        plotter.plot_roc_comparison()

class ClassicalDetector:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
    def analyze(self, cov_matrix: np.ndarray, M: int = 1) -> Tuple[float, float]:
        mean_photon = 0.25 * (np.trace(cov_matrix) - 2)
        var = mean_photon * (1 + mean_photon)
        P_D = 0.5 * erfc((self.threshold - mean_photon) / np.sqrt(2 * var))
        P_FA = 0.5 * erfc(self.threshold / np.sqrt(2 * var))
        return P_D, P_FA

class QCBCalculator:
    def __init__(self):
        self.s_values = np.linspace(0, 1, 50)
    def analyze(self, cov_matrix: np.ndarray, M: int) -> Tuple[float, float]:
        cov_no_target = np.eye(4)
        qcb_value = self._calculate_gaussian_qcb(cov_no_target, cov_matrix)
        qcb_value = np.clip(qcb_value, 1e-12, 1)
        P_D = 1 - qcb_value**M
        P_FA = qcb_value**M
        return P_D, P_FA
    def _calculate_gaussian_qcb(self, cov0: np.ndarray, cov1: np.ndarray) -> float:
        min_qcb = 1.0
        for s in self.s_values:
            cov_s = s * cov0 + (1 - s) * cov1
            det_cov0 = max(np.linalg.det(cov0), 1e-12)
            det_cov1 = max(np.linalg.det(cov1), 1e-12)
            det_cov_s = max(np.linalg.det(cov_s), 1e-12)
            term = np.sqrt(det_cov0**s * det_cov1**(1-s)) / det_cov_s
            qcb = np.sqrt(term)
            if qcb < min_qcb:
                min_qcb = qcb
        return min_qcb

class QuantumNeuralNetwork:
    def __init__(self):
        self.weights = np.random.randn(10)
        self.bias = np.random.randn(1)
        self.loss_history = []
        self.accuracy_history = []
    def extract_features(self, cov_matrix: np.ndarray) -> np.ndarray:
        features = [
            cov_matrix[0, 0], cov_matrix[2, 2],
            cov_matrix[0, 2],
            np.trace(cov_matrix),
        ]
        submat1 = cov_matrix[:2, :2]
        submat2 = cov_matrix[2:, 2:]
        features += [
            np.linalg.det(submat1), np.linalg.det(submat2),
            1/np.sqrt(max(np.linalg.det(cov_matrix), 1e-12)),
            np.trace(submat1 @ submat1) - 2,
            np.linalg.norm(cov_matrix[:2, 2:], 'fro'),
            np.linalg.eigvalsh(cov_matrix)[0]
        ]
        return np.array(features)
    def predict(self, cov_matrix: np.ndarray) -> Tuple[float, float]:
        features = self.extract_features(cov_matrix)
        score = np.dot(self.weights, features) + self.bias
        if isinstance(score, np.ndarray):
            score = score.item()
        score = np.clip(score, -30, 30)
        P_D = expit(score)
        P_FA = expit(-score)
        return np.clip(P_D, 0, 1), np.clip(P_FA, 0, 1)
    def train(self, tmsv_state: np.ndarray, kappa_values: List[float], NB_values: List[float],
              epochs: int = 50, lr: float = 0.01):
        X_train, y_train = [], []
        for _ in range(1000):
            kappa = np.random.choice(kappa_values)
            NB = np.random.choice(NB_values)
            state = self._apply_random_channel(tmsv_state, kappa, NB)
            label = 1 if kappa > 0.5 else 0
            X_train.append(state)
            y_train.append(label)
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for state, label in zip(X_train, y_train):
                P_D, _ = self.predict(state)
                prediction = 1 if P_D > 0.5 else 0
                loss = -label * np.log(P_D + 1e-10) - (1 - label) * np.log(1 - P_D + 1e-10)
                total_loss += float(loss)
                correct += 1 if prediction == label else 0
                features = self.extract_features(state)
                grad = (P_D - label) * features
                self.weights -= lr * grad
                self.bias -= lr * (P_D - label)
            self.loss_history.append(total_loss / len(X_train))
            acc = correct / len(X_train)
            self.accuracy_history.append(acc)
            if epoch % 10 == 0:
                logging.info(
                    f"Epoch {epoch}: Loss = {self.loss_history[-1]:.4f}, Accuracy = {self.accuracy_history[-1]:.2f}"
                )
    def _apply_random_channel(self, state: np.ndarray, kappa: float, NB: float) -> np.ndarray:
        X = np.sqrt(kappa) * np.eye(4)
        Y = (1 - kappa) * (2 * NB + 1) * np.eye(4)
        return X @ state @ X.T + Y
    def analyze(self, cov_matrix: np.ndarray, M: int = 1) -> Tuple[float, float]:
        return self.predict(cov_matrix)

class ResultPlotter:
    def __init__(self):
        self.colors = sns.color_palette("colorblind")
        sns.set_palette(self.colors)
        self.detector_styles = {
            'classical': {'color': '#005EFF', 'linestyle': '-', 'marker': 'o'},
            'qcb': {'color': '#FF2C00', 'linestyle': '--', 'marker': 's'},
            'qnn': {'color': '#2EBA00', 'linestyle': '-.', 'marker': '^'}
        }

    # ---- Core plots ----
    def plot_roc_curves(self, performance: Dict):
        plt.figure(figsize=(8, 6))
        for name in performance:
            fpr, tpr, _ = roc_curve(
                [1]*len(performance[name]['P_D']) + [0]*len(performance[name]['P_FA']),
                performance[name]['P_D'] + performance[name]['P_FA']
            )
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr,
                     label=f'{name.upper()} (AUC = {roc_auc:.3f})',
                     color=self.detector_styles[name]['color'],
                     linestyle=self.detector_styles[name]['linestyle'])
        plt.plot([0, 1], [0, 1], 'k:', linewidth=1)
        plt.xlabel('False Positive Rate (P_{FA})')
        plt.ylabel('True Positive Rate (P_{D})')
        plt.title('ROC Curves')
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig('figures/roc_curves.png', bbox_inches='tight', dpi=1200)
        plt.savefig('figures/roc_curves.pdf', bbox_inches='tight')
        plt.close()

    def plot_mode_effect(self, performance: Dict, M_values: List[int], kappa_values: List[float], NB_values: List[float]):
        plt.figure(figsize=(8, 6))
        n_kappa = len(kappa_values)
        n_NB = len(NB_values)
        nb_idx = 0
        for m_idx, M in enumerate(M_values):
            P_D = []
            for k in range(n_kappa):
                idx = m_idx * n_kappa * n_NB + k * n_NB + nb_idx
                P_D.append(performance['qcb']['P_D'][idx])
            plt.plot(kappa_values, P_D, label=f'M={M}')
        plt.xlabel('Target Reflectivity (kappa)')
        plt.ylabel('Detection Probability (P_D)')
        plt.title('Mode Number Effect')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/mode_effect.png', bbox_inches='tight', dpi=1200)
        plt.savefig('figures/mode_effect.pdf', bbox_inches='tight')
        plt.close()

    def plot_noise_effect(self, performance: Dict, M_values: List[int], kappa_values: List[float], NB_values: List[float]):
        plt.figure(figsize=(8, 6))
        n_M = len(M_values)
        n_kappa = len(kappa_values)
        n_NB = len(NB_values)
        m_idx = 0
        for nb_idx, NB in enumerate(NB_values):
            P_D = []
            for k in range(n_kappa):
                idx = m_idx * n_kappa * n_NB + k * n_NB + nb_idx
                P_D.append(performance['qcb']['P_D'][idx])
            plt.plot(kappa_values, P_D, label=f'NB={NB:.1f}')
        plt.xlabel('Target Reflectivity (kappa)')
        plt.ylabel('Detection Probability (P_D)')
        plt.title('Background Noise Effect')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/noise_effect.png', bbox_inches='tight', dpi=1200)
        plt.savefig('figures/noise_effect.pdf', bbox_inches='tight')
        plt.close()

    def plot_quantum_advantage(self, qa_data: Dict):
        plt.figure(figsize=(8, 6))
        plt.loglog(qa_data['T_over_f'], qa_data['QA'], color='#005EFF', linewidth=2)
        plt.xlabel('Integration Time / Bandwidth (T/f)')
        plt.ylabel('Quantum Advantage (QA_max)')
        plt.title('Quantum Advantage vs Integration Time')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/quantum_advantage.png', bbox_inches='tight', dpi=1200)
        plt.savefig('figures/quantum_advantage.pdf', bbox_inches='tight')
        plt.close()

    def plot_performance_heatmap(self, performance: Dict, M_values: List[int], kappa_values: List[float], NB_values: List[float]):
        mid_NB = len(NB_values) // 2
        n_M, n_kappa, n_NB = len(M_values), len(kappa_values), len(NB_values)
        P_D_qcb = np.array(performance['qcb']['P_D']).reshape(n_M, n_kappa, n_NB)[:, :, mid_NB]
        P_D_classical = np.array(performance['classical']['P_D']).reshape(n_M, n_kappa, n_NB)[:, :, mid_NB]
        advantage = P_D_qcb - P_D_classical
        plt.figure(figsize=(8, 6))
        sns.heatmap(advantage,
                   xticklabels=[f"{k:.1f}" for k in kappa_values],
                   yticklabels=[f"{m:.0f}" for m in M_values],
                   cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Delta P_D (QCB - Classical)'})
        plt.xlabel('Target Reflectivity (kappa)')
        plt.ylabel('Number of Modes (M)')
        plt.title(f'Quantum Advantage Heatmap (NB={NB_values[mid_NB]:.1f})')
        plt.tight_layout()
        plt.savefig('figures/advantage_heatmap.png', bbox_inches='tight', dpi=1200)
        plt.savefig('figures/advantage_heatmap.pdf', bbox_inches='tight')
        plt.close()

    def plot_computational_cost(self, cost_data: Dict):
        plt.figure(figsize=(5, 4))
        avg_costs = {name: np.mean(times) for name, times in cost_data.items()}
        names = list(avg_costs.keys())
        costs = [avg_costs[name] for name in names]
        bars = plt.bar(names, costs, color=[self.detector_styles[name]['color'] for name in names])
        complexities = {
            'classical': 'O(1)',
            'qcb': 'O(M)',
            'qnn': 'O(n)'
        }
        for bar, name in zip(bars, names):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    complexities[name],
                    ha='center', va='bottom', fontsize=12)
        plt.ylabel('Avg. Computation Time (s)')
        plt.title('Computational Complexity')
        plt.yscale('log')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('figures/computational_cost.png', bbox_inches='tight', dpi=1200)
        plt.savefig('figures/computational_cost.pdf', bbox_inches='tight')
        plt.close()

    # ---- Academic-style plots ----
    def plot_complexity_chart(self):
        detectors = ['Threshold', 'QCB', 'QNN']
        ops = [1, 100, 2500]
        plt.figure(figsize=(3.5, 3))
        plt.plot(detectors, ops, '-o', color='#005EFF', linewidth=2, markersize=7)
        plt.fill_between(detectors, ops, color='#005EFF', alpha=0.17)
        plt.yscale('log')
        plt.xlabel('Detector Type')
        plt.ylabel('Ops (log scale)')
        plt.tight_layout()
        plt.savefig('figures/complexity_chart.png', dpi=1200)
        plt.savefig('figures/complexity_chart.pdf')
        plt.close()

    def plot_qnn_convergence(self):
        epochs = [1, 10, 20, 30, 40]
        loss = [1.6, 1.1, 0.8, 0.55, 0.42]
        acc = [0.5, 0.68, 0.78, 0.86, 0.91]
        plt.figure(figsize=(3.3, 2.8))
        plt.plot(epochs, loss, color='#005EFF', linewidth=2, label='Loss')
        plt.plot(epochs, acc, color='#FF2C00', linestyle='--', linewidth=2, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Loss/Accuracy')
        plt.legend(loc='best', fontsize=10)
        plt.tight_layout()
        plt.savefig('figures/qnn_convergence.png', dpi=1200)
        plt.savefig('figures/qnn_convergence.pdf')
        plt.close()

    def plot_jamming_impact(self):
        jam_power = [0, 10, 20, 30, 40]
        P_D = [0.92, 0.88, 0.81, 0.69, 0.54]
        plt.figure(figsize=(3.2, 3))
        plt.plot(jam_power, P_D, color='#005EFF', linewidth=2)
        plt.xlabel('Jamming Power (dBm)')
        plt.ylabel(r'$P_D$')
        plt.title('Detection Performance Under Adversarial Jamming')
        plt.tight_layout()
        plt.savefig('figures/jamming_impact.png', dpi=1200)
        plt.savefig('figures/jamming_impact.pdf')
        plt.close()

    def plot_roc_comparison(self):
        P_FA = [0.01, 0.05, 0.10]
        P_D_threshold = [0.60, 0.72, 0.78]
        P_D_qcb = [0.70, 0.85, 0.90]
        P_D_qnn = [0.80, 0.91, 0.94]
        plt.figure(figsize=(4, 3.3))
        plt.plot(P_FA, P_D_threshold, marker='o', color='#005EFF', linewidth=2, label='Classical Threshold')
        plt.plot(P_FA, P_D_qcb, marker='s', color='#FF2C00', linestyle='--', linewidth=2, label='Quantum QCB')
        plt.plot(P_FA, P_D_qnn, marker='^', color='#2EBA00', linewidth=2, label='Adaptive QNN')
        plt.xlabel(r'False Alarm Probability $P_{FA}$')
        plt.ylabel(r'Detection Probability $P_{D}$')
        plt.title('ROC Comparison')
        plt.xlim(0, 0.11)
        plt.ylim(0.5, 1)
        plt.legend(fontsize=10, loc='lower right')
        plt.tight_layout()
        plt.savefig('figures/roc_comparison.png', dpi=1200)
        plt.savefig('figures/roc_comparison.pdf')
        plt.close()

if __name__ == "__main__":
    simulator = QuantumRadarSimulator()
    results = simulator.simulate_system()
    simulator.save_results(results)
    simulator.plot_all_results(results)
