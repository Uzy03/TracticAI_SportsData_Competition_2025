"""Tests for evaluation metrics.

This module tests various evaluation metrics including accuracy, F1, AUC, etc.
"""

import pytest
import torch
import numpy as np

from tacticai.modules.metrics import (
    TopKAccuracy,
    Accuracy,
    F1Score,
    AUC,
    ECE,
    CalibrationMetrics,
)


class TestTopKAccuracy:
    """Test top-k accuracy metric."""
    
    def test_top1_accuracy(self):
        """Test top-1 accuracy."""
        metric = TopKAccuracy(k=1)
        
        # Perfect predictions
        logits = torch.tensor([[10.0, 1.0, 2.0], [1.0, 10.0, 2.0], [1.0, 2.0, 10.0]])
        targets = torch.tensor([0, 1, 2])
        
        accuracy = metric(logits, targets)
        assert accuracy.item() == 1.0, f"Expected 1.0, got {accuracy.item()}"
        
        # All wrong predictions
        logits = torch.tensor([[1.0, 10.0, 2.0], [2.0, 1.0, 10.0], [10.0, 2.0, 1.0]])
        targets = torch.tensor([0, 1, 2])
        
        accuracy = metric(logits, targets)
        assert accuracy.item() == 0.0, f"Expected 0.0, got {accuracy.item()}"
    
    def test_top3_accuracy(self):
        """Test top-3 accuracy."""
        metric = TopKAccuracy(k=3)
        
        # All correct in top-3
        logits = torch.tensor([[10.0, 9.0, 8.0, 1.0], [9.0, 10.0, 8.0, 1.0]])
        targets = torch.tensor([0, 1])
        
        accuracy = metric(logits, targets)
        assert accuracy.item() == 1.0, f"Expected 1.0, got {accuracy.item()}"
        
        # Target not in top-3
        logits = torch.tensor([[1.0, 2.0, 3.0, 10.0], [1.0, 2.0, 3.0, 10.0]])
        targets = torch.tensor([0, 1])
        
        accuracy = metric(logits, targets)
        assert accuracy.item() == 0.0, f"Expected 0.0, got {accuracy.item()}"
    
    def test_topk_boundary_cases(self):
        """Test boundary cases for top-k accuracy."""
        metric = TopKAccuracy(k=2)
        
        # Single sample
        logits = torch.tensor([[10.0, 1.0, 2.0]])
        targets = torch.tensor([0])
        
        accuracy = metric(logits, targets)
        assert accuracy.item() == 1.0, f"Expected 1.0, got {accuracy.item()}"
        
        # k larger than number of classes
        metric = TopKAccuracy(k=10)
        logits = torch.tensor([[10.0, 1.0, 2.0]])
        targets = torch.tensor([0])
        
        accuracy = metric(logits, targets)
        assert accuracy.item() == 1.0, f"Expected 1.0, got {accuracy.item()}"
    
    def test_compute_all_k(self):
        """Test computing all k values."""
        metric = TopKAccuracy(k=1)
        
        logits = torch.tensor([[10.0, 1.0, 2.0], [1.0, 10.0, 2.0]])
        targets = torch.tensor([0, 1])
        
        results = metric.compute_all_k(logits, targets, k_values=[1, 2, 3])
        
        assert len(results) == 3
        assert 1 in results
        assert 2 in results
        assert 3 in results
        
        # All should be 1.0 for perfect predictions
        for k, acc in results.items():
            assert acc.item() == 1.0, f"Expected 1.0 for k={k}, got {acc.item()}"


class TestAccuracy:
    """Test accuracy metric."""
    
    def test_perfect_accuracy(self):
        """Test perfect accuracy."""
        metric = Accuracy()
        
        logits = torch.tensor([[10.0, 1.0, 2.0], [1.0, 10.0, 2.0], [1.0, 2.0, 10.0]])
        targets = torch.tensor([0, 1, 2])
        
        accuracy = metric(logits, targets)
        assert accuracy.item() == 1.0, f"Expected 1.0, got {accuracy.item()}"
    
    def test_zero_accuracy(self):
        """Test zero accuracy."""
        metric = Accuracy()
        
        logits = torch.tensor([[1.0, 10.0, 2.0], [2.0, 1.0, 10.0], [10.0, 2.0, 1.0]])
        targets = torch.tensor([0, 1, 2])
        
        accuracy = metric(logits, targets)
        assert accuracy.item() == 0.0, f"Expected 0.0, got {accuracy.item()}"
    
    def test_partial_accuracy(self):
        """Test partial accuracy."""
        metric = Accuracy()
        
        logits = torch.tensor([[10.0, 1.0, 2.0], [1.0, 10.0, 2.0], [1.0, 2.0, 10.0]])
        targets = torch.tensor([0, 0, 2])  # Only 2 out of 3 correct
        
        accuracy = metric(logits, targets)
        expected = 2.0 / 3.0
        assert abs(accuracy.item() - expected) < 1e-6, f"Expected {expected}, got {accuracy.item()}"


class TestF1Score:
    """Test F1 score metric."""
    
    def test_perfect_f1(self):
        """Test perfect F1 score."""
        metric = F1Score(average="weighted")
        
        logits = torch.tensor([[10.0, 1.0, 2.0], [1.0, 10.0, 2.0], [1.0, 2.0, 10.0]])
        targets = torch.tensor([0, 1, 2])
        
        f1 = metric(logits, targets)
        assert abs(f1.item() - 1.0) < 1e-6, f"Expected ~1.0, got {f1.item()}"
    
    def test_zero_f1(self):
        """Test zero F1 score."""
        metric = F1Score(average="weighted")
        
        logits = torch.tensor([[1.0, 10.0, 2.0], [2.0, 1.0, 10.0], [10.0, 2.0, 1.0]])
        targets = torch.tensor([0, 1, 2])
        
        f1 = metric(logits, targets)
        assert f1.item() == 0.0, f"Expected 0.0, got {f1.item()}"
    
    def test_per_class_f1(self):
        """Test per-class F1 scores."""
        metric = F1Score(average="none")
        
        logits = torch.tensor([[10.0, 1.0, 2.0], [1.0, 10.0, 2.0], [1.0, 2.0, 10.0]])
        targets = torch.tensor([0, 1, 2])
        
        f1_scores = metric(logits, targets)
        assert len(f1_scores) == 3
        for f1 in f1_scores:
            assert abs(f1.item() - 1.0) < 1e-6, f"Expected ~1.0, got {f1.item()}"


class TestAUC:
    """Test AUC metric."""
    
    def test_perfect_auc(self):
        """Test perfect AUC."""
        metric = AUC()
        
        # Perfect separation
        logits = torch.tensor([10.0, 10.0, -10.0, -10.0])
        targets = torch.tensor([1, 1, 0, 0])
        
        auc = metric(logits, targets)
        assert abs(auc.item() - 1.0) < 1e-6, f"Expected ~1.0, got {auc.item()}"
    
    def test_random_auc(self):
        """Test random AUC."""
        metric = AUC()
        
        # Random predictions
        logits = torch.tensor([0.0, 0.0, 0.0, 0.0])
        targets = torch.tensor([1, 1, 0, 0])
        
        auc = metric(logits, targets)
        # Should be around 0.5, but may vary due to ties
        assert 0.0 <= auc.item() <= 1.0, f"AUC should be in [0, 1], got {auc.item()}"
    
    def test_auc_with_pr(self):
        """Test AUC with precision-recall."""
        metric = AUC()
        
        logits = torch.tensor([10.0, 10.0, -10.0, -10.0])
        targets = torch.tensor([1, 1, 0, 0])
        
        auc_roc, auc_pr = metric(logits, targets, compute_auc_pr=True)
        
        assert abs(auc_roc.item() - 1.0) < 1e-6, f"Expected ~1.0 ROC, got {auc_roc.item()}"
        assert abs(auc_pr.item() - 1.0) < 1e-6, f"Expected ~1.0 PR, got {auc_pr.item()}"
    
    def test_auc_edge_cases(self):
        """Test AUC edge cases."""
        metric = AUC()
        
        # Single class
        logits = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1, 1, 1])
        
        # Should handle gracefully
        auc = metric(logits, targets)
        assert 0.0 <= auc.item() <= 1.0, f"AUC should be in [0, 1], got {auc.item()}"


class TestECE:
    """Test Expected Calibration Error."""
    
    def test_perfect_calibration(self):
        """Test perfectly calibrated predictions."""
        metric = ECE(n_bins=10)
        
        # Perfectly calibrated: confidence matches accuracy
        logits = torch.tensor([
            [2.0, 0.0, 0.0],  # High confidence, correct
            [2.0, 0.0, 0.0],  # High confidence, correct
            [0.0, 2.0, 0.0],  # High confidence, correct
            [0.0, 2.0, 0.0],  # High confidence, correct
        ])
        targets = torch.tensor([0, 0, 1, 1])
        
        ece = metric(logits, targets)
        assert ece.item() < 0.1, f"Expected low ECE, got {ece.item()}"
    
    def test_terrible_calibration(self):
        """Test terribly calibrated predictions."""
        metric = ECE(n_bins=10)
        
        # Terrible calibration: high confidence but wrong
        logits = torch.tensor([
            [2.0, 0.0, 0.0],  # High confidence, wrong
            [2.0, 0.0, 0.0],  # High confidence, wrong
            [0.0, 2.0, 0.0],  # High confidence, wrong
            [0.0, 2.0, 0.0],  # High confidence, wrong
        ])
        targets = torch.tensor([1, 1, 0, 0])  # All wrong
        
        ece = metric(logits, targets)
        assert ece.item() > 0.5, f"Expected high ECE, got {ece.item()}"
    
    def test_ece_bins(self):
        """Test ECE with different number of bins."""
        logits = torch.randn(100, 3)
        targets = torch.randint(0, 3, (100,))
        
        ece_5 = ECE(n_bins=5)(logits, targets)
        ece_10 = ECE(n_bins=10)(logits, targets)
        ece_20 = ECE(n_bins=20)(logits, targets)
        
        # All should be reasonable values
        assert 0.0 <= ece_5.item() <= 1.0
        assert 0.0 <= ece_10.item() <= 1.0
        assert 0.0 <= ece_20.item() <= 1.0


class TestCalibrationMetrics:
    """Test comprehensive calibration metrics."""
    
    def test_calibration_metrics(self):
        """Test calibration metrics."""
        metric = CalibrationMetrics(n_bins=10)
        
        # Create test data
        logits = torch.randn(100, 3)
        targets = torch.randint(0, 3, (100,))
        
        results = metric(logits, targets)
        
        assert "ece" in results
        assert "mce" in results
        
        assert 0.0 <= results["ece"].item() <= 1.0
        assert 0.0 <= results["mce"].item() <= 1.0
        
        # MCE should be >= ECE
        assert results["mce"].item() >= results["ece"].item()


class TestMetricConsistency:
    """Test metric consistency across different inputs."""
    
    def test_metric_determinism(self):
        """Test that metrics are deterministic."""
        metric = TopKAccuracy(k=3)
        
        logits = torch.tensor([[10.0, 1.0, 2.0, 3.0], [1.0, 10.0, 2.0, 3.0]])
        targets = torch.tensor([0, 1])
        
        # Multiple calls should give same result
        result1 = metric(logits, targets)
        result2 = metric(logits, targets)
        
        torch.testing.assert_close(result1, result2)
    
    def test_metric_device_consistency(self):
        """Test that metrics work on different devices."""
        if torch.cuda.is_available():
            metric = Accuracy()
            
            logits_cpu = torch.tensor([[10.0, 1.0, 2.0], [1.0, 10.0, 2.0]])
            targets_cpu = torch.tensor([0, 1])
            
            logits_cuda = logits_cpu.cuda()
            targets_cuda = targets_cpu.cuda()
            
            result_cpu = metric(logits_cpu, targets_cpu)
            result_cuda = metric(logits_cuda, targets_cuda)
            
            torch.testing.assert_close(result_cpu, result_cuda.cpu())


if __name__ == "__main__":
    pytest.main([__file__])
