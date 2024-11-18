"""
Density fitting MP2 on GPU
"""

import pyscf
import gpu4pyscf
import cupy as cp
import numpy as np

import gpu4pyscf.df
import gpu4pyscf.lib.logger
from gpu4pyscf.lib.cupy_helper import tag_array, get_avail_mem
from gpu4pyscf.mp.mp2 import MP2 as GPUMP2
import gpu4pyscf.df.int3c2e

from gpu4pyscf.mp.addons import (
    CONFIG_USE_SCF_WITH_DF,
    CONFIG_WITH_T2,
    CONFIG_WITH_CDERI_UOV,
    CONFIG_FP_TYPE,
    CONFIG_FP_TYPE_DECOMP,
    CONFIG_CDERI_ON_GPU,
    CONFIG_J2C_ALG,
)

MAX_BATCH_OCC = 256
""" Maximum batched occupation number in MP2. """


def kernel(mp, cderi_uov, occ_energy, vir_energy, with_t2=None, max_memory=None, verbose=None):
    """ DFMP2 (restricted) main function.

    Args:
        mp: gpu4pyscf.mp.dfmp2.DFMP2

        cderi_uov: np.ndarray or cp.ndarray
            Cholesky decomposed 3c-2e ERI (occ-vir part), in shape (aux, occ, vir) and c-contiguous.

        occ_energy: np.ndarray or list[np.ndarray]
            Occupied orbital energy levels.

        vir_energy: np.ndarray or list[np.ndarray]
            Virtual orbital energy levels.

        with_t2:
            Flag for generating MP2 amplitude. Recommended to be false if not for debugging.

        max_memory:
            Maximum memory for GPU.

        verbose: int or None

    Returns:
        dict

        e_corr_os: float
            Opposite-spin MP2 correlation energy

        e_corr_ss: float
            Same-spin MP2 correlation energy

        t2: cp.ndarray
            MP2 amplitude in (occ, occ, vir, vir); only available when ``with_t2 = True``.
    """
    verbose = mp.verbose if verbose is None else verbose
    log = gpu4pyscf.lib.logger.new_logger(mp, verbose)
    t0 = log.init_timer()

    with_t2 = mp.with_t2 if with_t2 is None else with_t2
    
    # sanity check
    assert isinstance(cderi_uov, (np.ndarray, cp.ndarray))
    naux, nocc, nvir = cderi_uov.shape
    assert nocc == occ_energy.size
    assert nvir == vir_energy.size
    
    # memory and type
    dtype = cderi_uov.dtype
    if with_t2:
        t2 = cp.empty((nocc, nocc, nvir, nvir), dtype=dtype)
    else:
        t2 = None

    max_memory = mp.mol.max_memory if max_memory is None else max_memory
    mem_avail = max(max_memory, get_avail_mem() / 1024**2)
    fp_size = 8 if dtype == np.float64 else 4
    fp_avail = 0.7 * mem_avail * 1024**2 / fp_size
    # minimum occ batch is hardcoded to 256
    batch_occ = min(fp_avail / (4 * naux * nvir), MAX_BATCH_OCC)
    log.debug(f"number of batched occupied orbitals: {batch_occ}")

    # preparation
    occ_energy = cp.asarray(occ_energy, dtype=dtype)
    vir_energy = cp.asarray(vir_energy, dtype=dtype)
    d_vv_gpu = - vir_energy[:, None] - vir_energy[None, :]

    # Note for efficiency
    # In GPU, arithmetic intensive matmul can overlap energy reduction,
    #   so python code here seems to be quite fast enough (cost of sum(t_ab * g_ab) may be overlapped by CUDA stream).
    #   This is not the same to numpy/CPU, since no replacement of CUDA stream on CPU.
    # Batch size (occupied) nither be too small (causes memory bandwidh consumption),
    #   nor too large (no prefetching of host-to-device transfer if cderi stored on CPU).
    stream = cp.cuda.get_current_stream()
    with stream:
        eng_bi1 = 0
        eng_bi2 = 0
        for ptr_i in range(0, nocc, batch_occ):
            log.debug(f"Load cderi step {ptr_i}/{nocc}")
            nbatch_i = min(batch_occ, nocc - ptr_i)
            cderi_uov_batch_i = cp.asarray(cderi_uov[:, ptr_i:ptr_i+nbatch_i])
            for ptr_j in range(0, ptr_i + nbatch_i, batch_occ):
                nbatch_j = min(batch_occ, ptr_i + nbatch_i - ptr_j)
                cderi_uov_batch_j = cp.asarray(cderi_uov[:, ptr_j:ptr_j+nbatch_j])
                for i in range(ptr_i, ptr_i + nbatch_i):
                    for j in range(ptr_j, min(ptr_j + nbatch_j, i + 1)):
                        factor = 2 if i != j else 1
                        g_ab = cderi_uov_batch_i[:, i-ptr_i].T @ cderi_uov_batch_j[:, j-ptr_j]
                        d_ab = occ_energy[i] + occ_energy[j] + d_vv_gpu
                        t_ab = g_ab / d_ab
                        eng_bi1 += factor * (t_ab * g_ab).sum()
                        eng_bi2 += factor * (t_ab.T * g_ab).sum()
                        if with_t2:
                            t2[i, j, :, :] = t_ab
                            t2[j, i, :, :] = t_ab.T

        eng_os = eng_bi1
        eng_ss = eng_bi1 - eng_bi2

    stream.synchronize()
    log.timer("kernel_cderi_on_cpu", *t0)

    result = {
        "e_corr_os": float(eng_os),
        "e_corr_ss": float(eng_ss),
    }
    if with_t2:
        result["t2"] = t2
    return result


def ao2mo(
        mp, auxmol=None, mo_coeff=None, mo_occ=None, frozen=None,
        cderi_on_gpu=None, with_cderi_uov=None,
        fp_type=None, fp_type_decomp=None, j2c_alg=None, verbose=None, max_memory=None):
    """ Wrapper function to generate Cholesky decomposed 3c-2e ERI.

    Args:
        mp: gpu4pyscf.mp.dfmp2.DFMP2
            Molecule object with normal basis set.

        auxmol: pyscf.gto.Mole
            Molecule object with auxiliary basis set.

        mo_coeff: np.ndarray
            Molecular orbital coefficients. Shape (nao, nmo).

        mo_occ: np.ndarray
            Molecular orbital occupation numbers. Shape (nmo).

        frozen: int or list(int) or None
            Frozen orbital indication.

        cderi_on_gpu: bool
            Whether generate and store Cholesky decomposed 3c-2e ERI on GPU.

        with_cderi_uov: bool
            Whether save Cholesky decomposed 3c-2e ERI in class attribute.

        fp_type: str
            Floating point type for final returned cderi_uov (step 4).

            - FP64: Double precision
            - FP32: Single precision

        fp_type_decomp: str
            Floating point for decompose 3c-2e ERI (step 3).

            - FP64: Double precision
            - FP32: Single precision

        j2c_alg: str
            Algorithm for decomposition.
            - "cd": Cholesky decomposition by default, eigen decomposition when scipy raises error
            - "eig": Eigen decomposition

        verbose: int or None

        max_memory: float or None
            Max memory in MB in CPU or GPU, depending on where Cholesky decomposed 3c-2e ERI is stored.
    """
    from gpu4pyscf.mp.addons import get_cderi_uov_direct_incore_cpu, get_cderi_uov_direct_incore_gpu

    # optional arguments
    mol = mp.mol
    auxmol = mp.with_df.auxmol if auxmol is None else auxmol
    cderi_on_gpu = mp.cderi_on_gpu if cderi_on_gpu is None else cderi_on_gpu
    fp_type = mp.fp_type if fp_type is None else fp_type
    fp_type_decomp = mp.fp_type_decomp if fp_type_decomp is None else fp_type_decomp
    j2c_alg = mp.j2c_alg if j2c_alg is None else j2c_alg
    verbose = mp.mol.verbose if verbose is None else verbose

    _, occ_coeff, vir_coeff, _ = mp.split_mo_coeff(mo_coeff, frozen=frozen, mo_occ=mo_occ)

    if cderi_on_gpu is False:
        if isinstance(occ_coeff, cp.ndarray):
            occ_coeff = occ_coeff.get()
            vir_coeff = vir_coeff.get()
        cderi_result = get_cderi_uov_direct_incore_cpu(
            mol, auxmol, occ_coeff, vir_coeff,
            fp_type=fp_type, fp_type_decomp=fp_type_decomp, j2c_alg=j2c_alg, verbose=verbose, max_memory=max_memory)
        mp.j2c_decomp = cderi_result["j2c_decomp"]
    elif cderi_on_gpu is True:
        occ_coeff = cp.asarray(occ_coeff)
        vir_coeff = cp.asarray(vir_coeff)
        cderi_result = get_cderi_uov_direct_incore_gpu(
            mol, auxmol, occ_coeff, vir_coeff,
            fp_type=fp_type, fp_type_decomp=fp_type_decomp, j2c_alg=j2c_alg, verbose=verbose, max_memory=max_memory)
        mp.j2c_decomp = cderi_result["j2c_decomp"]
        mp.intopt = cderi_result["intopt"]
    else:
        raise NotImplementedError
    if with_cderi_uov:
        mp.cderi_uov = cderi_result["cderi_uov"]
    return cderi_result["cderi_uov"]


class DFMP2(GPUMP2):
    _keys = {
        "with_df", "auxbasis",
        "with_t2", "cderi_on_gpu", "with_cderi_uov", "fp_type_decomp", "mo_energy", "fp_type", "j2c_alg",
        "j2c_decomp", "intopt", "t2", "cderi_uov"
    }

    def __init__(self, mf, frozen=None, auxbasis=None, mo_coeff=None, mo_occ=None, mo_energy=None):
        super().__init__(mf, frozen, mo_coeff, mo_occ)
        if not mf.converged:
            raise RuntimeError("SCF must be converged to perform DFMP2. Non-canonical DFMP2 currently not implemented.")
        self.mo_energy = mf.mo_energy if mo_energy is None else mo_energy

        # auxiliary and cderi configuration
        if auxbasis is not None:
            self.with_df = gpu4pyscf.df.DF(mf.mol, auxbasis=auxbasis)
            self.with_df.auxmol = gpu4pyscf.df.make_auxmol(mf.mol, self.with_df.auxbasis)
        elif getattr(mf, 'with_df', None) and CONFIG_USE_SCF_WITH_DF:
            self.with_df = mf.with_df
        else:
            self.with_df = gpu4pyscf.df.DF(mf.mol)
            self.with_df.auxbasis = gpu4pyscf.df.make_auxbasis(mf.mol, mp2fit=True)
            self.with_df.auxmol = gpu4pyscf.df.make_auxmol(mf.mol, self.with_df.auxbasis)

        # configuration keywords
        self.with_t2 = CONFIG_WITH_T2
        self.fp_type = CONFIG_FP_TYPE
        self.fp_type_decomp = CONFIG_FP_TYPE_DECOMP
        self.with_cderi_uov = CONFIG_WITH_CDERI_UOV
        self.cderi_on_gpu = CONFIG_CDERI_ON_GPU
        self.j2c_alg = CONFIG_J2C_ALG

        # output intermediates or results, should not be modified by user
        self.j2c_decomp = NotImplemented  # type: dict
        self.intopt = NotImplemented  # type: gpu4pyscf.df.int3c2e.VHFOpt
        self.t2 = NotImplemented  # type: cp.ndarray
        self.cderi_uov = NotImplemented  # type: cp.ndarray
        self.e_corr = NotImplemented  # type: float
        self.e_corr_os = NotImplemented  # type: float
        self.e_corr_ss = NotImplemented  # type: float

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    def mo_splitter(self, frozen=None, mo_occ=None):
        from gpu4pyscf.mp.addons import mo_splitter_restricted
        return mo_splitter_restricted(self, frozen=frozen, mo_occ=mo_occ)

    def get_frozen_mask(self, frozen=None, mo_occ=None):
        from gpu4pyscf.mp.addons import get_frozen_mask_restricted
        return get_frozen_mask_restricted(self, frozen=frozen, mo_occ=mo_occ)

    def split_mo_coeff(self, mo_coeff=None, frozen=None, mo_occ=None):
        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        masks = self.mo_splitter(frozen=frozen, mo_occ=mo_occ)
        return [mo_coeff[:, mask] for mask in masks]

    def split_mo_energy(self, mo_energy=None, frozen=None, mo_occ=None):
        mo_energy = self.mo_energy if mo_energy is None else mo_energy
        masks = self.mo_splitter(frozen=frozen, mo_occ=mo_occ)
        return [mo_energy[mask] for mask in masks]

    def kernel(
            self, frozen=None, mo_occ=None, mo_energy=None, mo_coeff=None, eris=None, with_t2=None,
            max_memory=None, verbose=None):
        log = gpu4pyscf.lib.logger.new_logger(self, verbose)
        t0 = log.init_timer()

        with_t2 = self.with_t2 if with_t2 is None else with_t2

        if self.verbose >= gpu4pyscf.lib.logger.WARN:
            self.check_sanity()
        self.dump_flags()

        self.e_hf = self.get_e_hf(mo_coeff=mo_coeff)

        if eris is None:
            eris = self.ao2mo()

        _, occ_energy, vir_energy, _ = self.split_mo_energy(mo_energy, frozen=frozen, mo_occ=mo_occ)
        result = kernel(self, eris, occ_energy, vir_energy, max_memory, verbose)

        e_corr_os = result["e_corr_os"]
        e_corr_ss = result["e_corr_ss"]
        e_corr = tag_array(cp.asarray(e_corr_os + e_corr_ss), e_corr_os=e_corr_os, e_corr_ss=e_corr_ss)

        if with_t2:
            self.t2 = result["t2"]

        self.e_corr = e_corr
        self.e_corr_os = e_corr_os
        self.e_corr_ss = e_corr_ss

        log.timer("kernel of DFMP2", *t0)
        return e_corr

    ao2mo = ao2mo



if __name__ == "__main__":
    def mf_on_cpu():
        import pyscf.mp
        import pyscf.df

        pyscf.lib.logger.TIMER_LEVEL = 1
        gpu4pyscf.lib.logger.TIMER_LEVEL = 1

        e_ref = -0.271121778503736

        mol = pyscf.gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP").build()
        mf = pyscf.scf.RHF(mol).density_fit(auxbasis="def2-universal-jkfit").run()
        mp_cpu = pyscf.mp.dfmp2.DFMP2(mf)
        mp_cpu.with_df = pyscf.df.DF(mol, auxbasis="def2-TZVP-ri").build()
        mp_cpu.run()
        assert np.isclose(mp_cpu.e_corr, e_ref)

        # cderi on gpu
        mf = mf.to_gpu()
        mp = DFMP2(mf).run()
        assert np.isclose(mp.e_corr, e_ref)

        # cderi on cpu
        # function calls can be validated in output
        mp = DFMP2(mf).run(cderi_on_gpu=False)
        assert np.isclose(mp.e_corr, e_ref)

    def mf_on_gpu():

        pyscf.lib.logger.TIMER_LEVEL = 1
        gpu4pyscf.lib.logger.TIMER_LEVEL = 1

        e_ref = -0.271121778503736

        mol = pyscf.gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP").build()
        mf = gpu4pyscf.scf.RHF(mol).density_fit(auxbasis="def2-universal-jkfit").run()

        # cderi on gpu
        mf = mf.to_gpu()
        mp = DFMP2(mf).run()
        assert np.isclose(mp.e_corr, e_ref)

        # cderi on cpu
        # function calls can be validated in output
        mp = DFMP2(mf).run(cderi_on_gpu=False)
        assert np.isclose(mp.e_corr, e_ref)

    mf_on_cpu()
    mf_on_gpu()

