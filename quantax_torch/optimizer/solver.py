import torch


def _get_eigs_inv(vals, tol, atol):
    if tol is None:
        tol = {torch.float64: 1e-12, torch.float32: 1e-6, torch.float16: 1e-3}[vals.dtype]
    magnitudes = vals.abs()
    cutoff = tol * magnitudes.max() + atol
    safe = torch.where(magnitudes > 0, magnitudes, 1.0)
    soft = 1 / (1 + (cutoff / safe).pow(6))
    return torch.where(magnitudes > 0, vals.sign() * soft / safe, 0.0)


def _sum_without_noise(inputs, tol_snr):
    total = inputs.sum(dim=0)
    if tol_snr <= 1e-6:
        return total
    mean = inputs.mean(dim=0)
    error = ((inputs - mean).abs().square().mean(dim=0) / len(inputs)).sqrt()
    abs_mean = mean.abs()
    ratio = tol_snr * error / torch.where(abs_mean > 0, abs_mean, 1.0)
    return torch.where(abs_mean > 0, total / (1 + ratio.pow(6)), 0.0)


def minsr_pinv_eig(tol=None, atol=0.0, tol_snr=0.0):
    def solve(T, b):
        if not bool(T.isfinite().all() & b.isfinite().all()):
            raise FloatingPointError("Non-finite matrix or right-hand side in the MinSR solve.")
        vals, U = torch.linalg.eigh(T)
        inverse = _get_eigs_inv(vals, tol, atol)
        rhs = U.mH @ b if tol_snr <= 1e-6 else _sum_without_noise(U.conj() * b[:, None], tol_snr)
        return U @ (inverse * rhs)

    return solve


def pinvh_solve(tol=None, atol=0.0):
    return minsr_pinv_eig(tol, atol)


def minnorm_pinv_eig(tol=None, atol=0.0, tol_snr=0.0):
    solve = minsr_pinv_eig(tol, atol, tol_snr)
    return lambda A, b: A.mH @ solve(A @ A.mH, b)


def lstsq_pinv_eig(tol=None, atol=0.0, tol_snr=0.0):
    def solve(A, b):
        vals, V = torch.linalg.eigh(A.mH @ A)
        inv = _get_eigs_inv(vals, tol, atol)
        rhs = (A @ V).conj() * b[:, None]
        return V @ (inv * _sum_without_noise(rhs, tol_snr))

    return solve


def auto_pinv_eig(tol=None, atol=0.0, tol_snr=0.0):
    minnorm = minnorm_pinv_eig(tol, atol, tol_snr)
    lstsq = lstsq_pinv_eig(tol, atol, tol_snr)

    def solve(A, b):
        return minnorm(A, b) if A.shape[0] < A.shape[1] else lstsq(A, b)

    return solve


def minnorm_shift_eig(ashift=1e-4):
    def solve(A, b):
        T = A @ A.mH
        T.diagonal().add_(ashift)
        return A.mH @ torch.linalg.solve(T, b)

    return solve


class lstsq_shift_cg:
    def __init__(self, diag_shift=0.01, tol=1e-5, atol=0.0, maxiter=None):
        self.diag_shift = diag_shift
        self.tol = tol
        self.atol = atol
        self.maxiter = maxiter

    def __call__(self, A, b):
        diagonal = self.diag_shift * A.abs().square().sum(dim=0)
        rhs = A.mH @ b
        x = torch.zeros_like(rhs)
        r = p = rhs
        criterion = max(self.atol, self.tol * r.norm().item())
        rr = torch.vdot(r, r).real
        for _ in range(rhs.numel() * 10 if self.maxiter is None else self.maxiter):
            if rr <= criterion**2:
                break
            Ap = A.mH @ (A @ p) + diagonal * p
            curvature = torch.vdot(p, Ap).real
            if curvature <= 0:
                break
            alpha = rr / curvature
            x = x + alpha * p
            r = r - alpha * Ap
            next_rr = torch.vdot(r, r).real
            p = r + (next_rr / rr) * p
            rr = next_rr
        return x, None


def sgd_solver():
    return lambda A, b: (A.mH @ b) / b.shape[0]
