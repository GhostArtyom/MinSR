import torch

from .solver import auto_pinv_eig, pinvh_solve
from ..state import DenseState
from ..utils.spins import ints_to_array


def _stack_components(x):
    return torch.cat((x.real, x.imag if x.is_complex() else torch.zeros_like(x)), dim=0)


class TDVP:
    def __init__(self, state, hamiltonian, imag_time=True, solver=None, use_kazcmarz=False):
        self.state = state
        self.hamiltonian = hamiltonian
        self.imag_time = imag_time
        self.energy = None
        self.VarE = None
        self._solver = auto_pinv_eig() if solver is None else solver
        self._Omean = None
        self._solver_info = None
        self._last_step = None
        self._use_kazcmarz = use_kazcmarz

    @property
    def vs_type(self):
        return self.state.vs_type

    def get_Ebar(self, samples):
        Eloc = self.hamiltonian.Oloc(self.state, samples).to(self.state.dtype)
        weight = samples.reweight_factor
        Emean = (Eloc * weight).mean()
        self.energy = Emean.real
        self.VarE = ((Eloc - Emean).abs().square() * weight).mean()
        return (Eloc - Eloc.mean()) * (weight / samples.nsamples).sqrt()

    def _center_jacobian(self, jac, weight):
        return (jac - jac.mean(dim=0)) * (weight[:, None] / len(weight)).sqrt()

    def get_Obar(self, samples):
        jac = self.state.jacobian(samples.spins)
        self._Omean = (jac * samples.reweight_factor[:, None]).mean(dim=0)
        return self._center_jacobian(jac, samples.reweight_factor)

    def _solve(self, matrix, rhs):
        result = self._solver(matrix, rhs)
        if isinstance(result, tuple):
            result, self._solver_info = result
        return result

    def solve(self, Obar, Ebar):
        if not self.imag_time:
            Ebar = Ebar * 1j
        if self.vs_type in (1, 2):
            Obar, Ebar = _stack_components(Obar), _stack_components(Ebar)
        return self._solve(Obar, Ebar)

    def build_step(self, step, Omean, Emean):
        dtype = torch.promote_types(step.dtype, Omean.dtype)
        step0 = -torch.dot(Omean.to(dtype), step.to(dtype))
        if not self.imag_time:
            step0 = 1j * Emean + step0
        if self.vs_type == 1:
            step = torch.complex(*step.chunk(2))
        return torch.cat([step0.reshape(1), step]).to(self.state.dtype)

    @torch.no_grad()
    def get_step(self, samples):
        Ebar = self.get_Ebar(samples)
        Obar = self.get_Obar(samples)
        if self._use_kazcmarz and self._last_step is not None:
            residual = Ebar - Obar @ self._last_step.to(Obar.dtype)
            step = self.solve(Obar, residual) + self._last_step
        else:
            step = self.solve(Obar, Ebar)
        if self._use_kazcmarz:
            self._last_step = step
        return self.build_step(step, self._Omean, self.energy)


class TDVP_exact(TDVP):
    def __init__(self, state, hamiltonian, imag_time=True, solver=None, symm=None):
        super().__init__(state, hamiltonian, imag_time, solver)
        self.symm = state.symm if symm is None else symm
        basis = self.symm.basis
        basis.make()
        self.spins = ints_to_array(basis.states, state.nsites)
        factors = basis.get_amp(basis.states.copy())
        if not state.dtype.is_complex:
            factors = factors.real
        self._factors = torch.as_tensor(factors, device=state.device)

    @torch.no_grad()
    def get_step(self):
        wf = self.state(self.spins) / self._factors
        psi = DenseState(wf, self.symm)
        Hpsi = (self.hamiltonian @ psi).wave_function
        norm = wf.norm()
        E = (wf.conj() * Hpsi).sum() / norm.square()
        epsilon = (Hpsi - E * wf) / norm
        self.energy = E.real
        self.VarE = epsilon.abs().square().sum()
        jac = self.state.jacobian(self.spins, logarithmic=False) / self._factors[:, None]
        self._Omean = (wf.conj()[:, None] * jac).sum(dim=0) / norm.square()
        Obar = (jac - wf[:, None] * self._Omean) / norm
        step = self.solve(Obar, epsilon)
        return self.build_step(step, self._Omean, self.energy)


class TimeEvol(TDVP):
    def __init__(self, state, hamiltonian, solver=None, max_parallel=None):
        super().__init__(state, hamiltonian, imag_time=False, solver=pinvh_solve() if solver is None else solver)
        self.max_parallel = max_parallel

    def get_SF(self, samples):
        Ebar = self.get_Ebar(samples)
        if self.max_parallel is None or samples.nsamples <= self.max_parallel:
            Obar = self.get_Obar(samples)
            return Obar.mH @ Obar, Obar.mH @ Ebar
        size = self.state.nparams
        if self.vs_type == 1:
            size *= 2
        dtype = self.state.dtype
        device = self.state.device
        S = torch.zeros((size, size), dtype=dtype, device=device)
        F = torch.zeros(size, dtype=dtype, device=device)
        total = torch.zeros_like(F)
        for spins, E in zip(samples.spins.split(self.max_parallel), Ebar.split(self.max_parallel)):
            jac = self.state.jacobian(spins)
            total += jac.sum(dim=0)
            S += jac.mH @ jac
            F += jac.mH @ E
        self._Omean = total / samples.nsamples
        return S / samples.nsamples - torch.outer(self._Omean.conj(), self._Omean), F / samples.nsamples**0.5

    @torch.no_grad()
    def get_step(self, samples):
        if not torch.allclose(samples.reweight_factor, torch.ones_like(samples.reweight_factor)):
            raise ValueError("TimeEvol requires unweighted samples.")
        S, F = self.get_SF(samples)
        step = self._solve(S, 1j * F) if self.vs_type == 0 else self._solve(S.real, -F.imag)
        return self.build_step(step, self._Omean, self.energy)
