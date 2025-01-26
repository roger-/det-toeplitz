import numpy as np
from scipy.linalg import toeplitz
from toeplitz import slogdet_toeplitz, log_det_toeplitz, det_toeplitz, ToeplitzDeterminant


def test_all_toeplitz_functions():
    test_cases = [
        ("Real symmetric positive", np.array([2.0, 0.5, 0.1])),
        ("Real symmetric negative", np.array([-2.0, 0.5, 0.1])),
        ("Real symmetric indefinite", np.array([1.0, 2.0, 0.5])),
        ("Complex hermitian positive", np.array([2.0, 0.5-0.1j, 0.1+0.2j])),
        ("Complex hermitian indefinite", np.array([2.0, 1.5-0.1j, 0.9+0.2j])),
        ("Real non-symmetric positive", (np.array([2.0, 0.5, 0.1]), np.array([2.0, 0.3, 0.2]))),
        ("Real non-symmetric negative", (np.array([-2.0, 0.5, 0.1]), np.array([-2.0, -0.3, 0.2]))),
        ("Complex non-symmetric", (np.array([2.0, 0.5-0.1j, 0.1+0.2j]), 
                                 np.array([2.0, 0.3+0.1j, 0.2-0.1j]))),
        # ("Near singular", np.array([1e-15, 1.0, 1.0])),
        ("Large dimension symmetric", np.array([2.0] + [0.1]*9)),
        ("Alternating signs", np.array([1.0, -0.5, 0.25, -0.125])),
        # ("Highly ill-conditioned", np.array([1e-10, 1e5, 1e-5])),
        # ("Pure rotation", np.array([1.0, np.exp(1j*np.pi/4), np.exp(1j*np.pi/2)])),
        ("All zeros except diagonal", np.array([1.0, 0.0, 0.0, 0.0]))
    ]
    
    tol = 1e-10
    results = {
        'slogdet': True,
        'logdet': True,
        'det': True
    }
    
    for desc, case in test_cases:
        print(f"\n=== {desc} ===")
        
        if isinstance(case, tuple):
            c, r = case
            T = toeplitz(c, r)
        else:
            c = case
            T = toeplitz(c, np.conj(c))
            
        det_direct = np.linalg.det(T)
        
        # Test slogdet_toeplitz
        res_slog = slogdet_toeplitz(case)
        det_slog = res_slog.sign * np.exp(res_slog.logabsdet)
        
        # Test log_det_toeplitz (only for positive determinants)
        if det_direct > 0:
            try:
                logdet = log_det_toeplitz(case)
                det_log = np.exp(logdet)
                passed_log = abs(det_direct - det_log)/abs(det_direct) < tol
                print(f"log_det_toeplitz: {'PASS' if passed_log else 'FAIL'}")
                results['logdet'] &= passed_log
            except Exception as e:
                print(f"log_det_toeplitz: FAIL")
                results['logdet'] = False
        
        # Test det_toeplitz
        try:
            det_direct_func = det_toeplitz(case)
            passed_det = abs(det_direct - det_direct_func)/abs(det_direct) < tol
            print(f"det_toeplitz: {'PASS' if passed_det else 'FAIL'}")
            results['det'] &= passed_det
        except Exception as e:
            print(f"det_toeplitz: FAIL")
            results['det'] = False
        
        # Check slogdet results
        if abs(det_direct) < tol:
            passed_slog = res_slog.sign == 0 and np.isneginf(res_slog.logabsdet)
        else:
            rel_err_slog = abs(det_direct - det_slog)/abs(det_direct)
            passed_slog = rel_err_slog < tol and (abs(abs(res_slog.sign) - 1.0) < tol or res_slog.sign == 0)
            
        print(f"slogdet_toeplitz: {'PASS' if passed_slog else 'FAIL'}")
        results['slogdet'] &= passed_slog
    
    print("\n=== Overall Results ===")
    for func, passed in results.items():
        print(f"{func}: {'PASS' if passed else 'FAIL'}")
    
    return all(results.values())

if __name__ == "__main__":
    test_all_toeplitz_functions()