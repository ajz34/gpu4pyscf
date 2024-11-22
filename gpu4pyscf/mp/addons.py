"""
GPU4PySCF config and utilities for MP2 module
"""
import cupy.cuda
import pyscf
import gpu4pyscf
import numpy as np
import cupy as cp

from gpu4pyscf.lib.cupy_helper import get_avail_mem as get_avail_gpu_mem

import cupyx.scipy.linalg
import gpu4pyscf.lib.logger
import gpu4pyscf.df.int3c2e
import pyscf.df.incore
from pyscf import __config__

# region GPU4PySCF config

CONFIG_USE_SCF_WITH_DF = getattr(__config__, "gpu_mp_dfmp2_use_scf_with_df", False)
""" Flag for using cderi from SCF object (not implemented).

This option will be overrided if auxiliary basis set is explicitly specified.

- True: Use cderi from SCF object. This will override user-specified auxiliary basis.
- False: Always generate cderi.
"""

CONFIG_WITH_T2 = getattr(__config__, "gpu_mp_dfmp2_with_t2", False)
""" Flag for computing T2 amplitude.

In many cases, this is not recommended, except for debugging.
Energy (or possibly gradient in future) can be computed without T2 amplitude.
"""

CONFIG_WITH_CDERI_OVL = getattr(__config__, "gpu_mp_dfmp2_with_cderi_ovl", False)
""" Flag for save Cholesky decomposed 3c-2e ERI (occ-vir part). """

CONFIG_FP_TYPE = getattr(__config__, "gpu_mp_dfmp2_fp_type", "FP64")
""" Floating point type for MP2 calculation.

Currently only FP64 and FP32 are supported.
To use TF32, set environment variable ``CUPY_TF32=1`` before running python / importing cupy, and set ``FP32`` for this option.
Use TF32 with caution for RI-MP2. TF32 is not recommended when performing LT-OS-MP2.

- FP64: Double precision
- FP32: Single precision
"""

CONFIG_FP_TYPE_DECOMP = getattr(__config__, "gpu_mp_dfmp2_same_fp_type_decomp", None)
""" Flag for using the same floating point type for decomposition.

Note that ERI is always generated in FP64. This only affects the decomposition.

- None: Use the same floating point type as the MP2 calculation.
- FP64: Double precision
- FP32: Single precision
"""

CONFIG_CDERI_ON_GPU = getattr(__config__, "gpu_mp_dfmp2_cderi_on_gpu", True)
""" Flag for storing cderi (MO part) on GPU.

- None: (not implemented) Automatically choose based on the available GPU memory.
- True: Always storing cderi on GPU DRAM.
- False: Always storing cderi on CPU DRAM.
"""

CONFIG_J2C_ALG = getattr(__config__, "gpu_mp_dfmp2_j2c_alg", "cd")
""" Algorithm for j2c decomposition.

- "cd": Cholesky decomposition
- "eig": Eigen decomposition
"""

CONFIG_THRESH_LINDEP = getattr(__config__, "mp_dfmp2_thresh_lindep", 1e-10)
""" Threshold for linear dependence detection of j2c. """

MIN_BATCH_AUX_CPU = 32
MIN_BATCH_AUX_GPU = 32
BLKSIZE_AO = 128
CUTOFF_J3C = 1e-10

# endregion


# region utilities of post-SCF

def get_frozen_mask_restricted(mp, frozen=None, mo_occ=None):
    """ Get boolean mask for the restricted reference orbitals.

    This will return numpy object, instead of cupy object.

    Args:
        mp: pyscf.lib.StreamObject

        frozen: int or list(int) or None

            - int: number of frozen occupied orbitals
            - list: frozen orbital indices

        mo_occ: np.ndarray
            Molecular occupation numbers

    Returns:
        moidx: np.ndarray
            Mask array of frozen (true) and active (false) orbitals.

    See also:
        pyscf.mp.mp2.get_frozen_mask
    """
    mo_occ = mp.mo_occ if mo_occ is None else mo_occ
    frozen = mp.frozen if frozen is None else frozen

    moidx = np.ones(mo_occ.size, dtype=bool)
    if hasattr(mp, "_nmo") and mp._nmo is not None:
        # frozen virtual orbitals by number
        moidx[mp._nmo:] = False
    if frozen is None:
        pass
    elif isinstance(frozen, (int, np.integer, cp.integer)):
        # frozen occupied orbitals by number
        moidx[:int(frozen)] = False
    elif len(frozen) > 0:
        # frozen orbitals by index list
        moidx[list(frozen)] = False
    else:
        raise NotImplementedError
    return moidx


def mo_splitter_restricted(mp, frozen=None, mo_occ=None):
    """ Active orbital masks for the restricted reference orbitals.

    Args:
        mp: pyscf.lib.StreamObject

        frozen: int or list(int) or None

            - int: number of frozen occupied orbitals
            - list: frozen orbital indices

        mo_occ: np.ndarray
            Molecular occupation numbers.

    Returns:
        masks: list(np.ndarray)

            - occupied frozen
            - occupied active
            - virtual active
            - virtual frozen
    """
    mo_occ = mp.mo_occ if mo_occ is None else mo_occ
    frozen = mp.frozen if frozen is None else frozen
    if isinstance(mo_occ, cp.ndarray):
        mo_occ = mo_occ.get()
    mask_act = get_frozen_mask_restricted(mp, mo_occ=mo_occ, frozen=frozen)
    mask_occ = mo_occ > 1e-6
    masks = [
        mask_occ  & ~mask_act,    # frz occ
        mask_occ  &  mask_act,    # act occ
        ~mask_occ &  mask_act,    # act vir
        ~mask_occ & ~mask_act,    # frz vir
    ]
    return masks

# endregion


# region ERI on GPU

def get_dtype(type_token, is_gpu):
    if type_token.upper() == "FP64":
        return cp.float64 if is_gpu else np.float64
    elif type_token.upper() == "FP32":
        return cp.float32 if is_gpu else np.float32
    else:
        raise ValueError(f"Unknown type {type_token}")


def get_j2c_decomp_gpu(streamobj, j2c, alg=CONFIG_J2C_ALG, thresh_lindep=CONFIG_THRESH_LINDEP, verbose=None):
    r""" Get j2c decomposition in GPU.

    Given 2c-2e ERI (j2c) :math:`J_{PQ}`, decomposed j2c :math:`L_{PQ}` is defined as

    .. math::
        \sum_{R} L_{PR} L_{QR} = J_{PQ}

    This decomposition can be obtained by Cholesky decomposition or eigen decomposition.

    Args:
        streamobj: pyscf.lib.StreamObject

        j2c: cp.ndarray
            2c-2e ERI, could be obtained from ``mol.intor("int2c2e")`` or other equilvants.

        alg: str
            Algorithm for decomposition.
            - "cd": Cholesky decomposition by default, eigen decomposition when scipy raises error
            - "eig": Eigen decomposition

        thresh_lindep: float
            Threshold for linear dependence detection of j2c.

        verbose: int

    Returns:
        dict

        j2c_l: cp.ndarray
            Decomposed j2c. Shape (naux, naux).

        j2c_l_inv: cp.ndarray
            Matrix inverse of ``j2c_l``. Only computed when algorithm is ``eig``.

        tag: str
            Algorithm for decomposition.

            - "cd": Cholesky decomposition
            - "eig": Eigen decomposition

    See also:
        get_j2c_decomp_cpu
    """
    log = gpu4pyscf.lib.logger.new_logger(streamobj, verbose)
    t0 = log.init_timer()

    # Cholesky decomposition
    if alg.lower().startswith("cd"):
        log.debug("j2c decomposition by Cholesky decomposition")
        j2c_l = cp.linalg.cholesky(j2c)
        if not cp.isnan(j2c_l[0, 0]):
            # cupy does not raise error, but will give nan lower triangular on return
            log.timer("get_j2c_decomp by cd", *t0)
            return {
                "j2c_l": j2c_l,
                "tag": "cd",
            }
        else:
            log.warn("j2c decomposition by Cholesky failed. Switching to eigen decomposition.")
            alg = "eig"

    # Eigen decomposition
    if alg.lower().startswith("eig"):
        log.debug("j2c decomposition by eigen")
        e, u = cp.linalg.eigh(j2c)
        cond = abs(e).max() / abs(e).min()
        keep = e > thresh_lindep
        rkeep = e < - thresh_lindep
        if rkeep.sum() > 0:
            log.warn(f"Some {rkeep.sum()} j2c eigenvalues are much smaller than zero, which is unexpected.")
        log.debug(f"cond(j2c) = {cond}")
        log.debug(f"keep {keep.sum()}/{keep.size} j2c vectors")
        e = e[keep]
        u = u[:, keep]
        j2c_l = u * e**0.5 @ u.T.conj()
        j2c_l_inv = u * e**-0.5 @ u.T.conj()
        log.timer("get_j2c_decomp by eig", *t0)
        return {
            "j2c_l": j2c_l,
            "j2c_l_inv": j2c_l_inv,
            "tag": "eig",
        }
    else:
        raise ValueError(f"Unknown j2c decomposition algorithm: {alg}")


def get_int3c2e_by_aux_id(mol, intopt, idx_k, omega=None, out=None):
    """ Generator of int3c2e on GPU.

    This function only give 3-dimension ``int3c2e`` (k, j, i) in c-contiguous array.
    Currently, other integrals are not available.

    Args:
        mol: pyscf.gto.Mole
            Molecule object with normal basis set.

        intopt: gpu4pyscf.df.int3c2e.VHFOpt
            Integral optimizer object for 3c-2e ERI on GPU.

        idx_k: int
            Index of third index in 3c-2e ERI (most cases auxiliary basis).

        omega: float or None
            Range separate parameter.

        out: cp.ndarray

    Returns:
        cp.ndarray

    Example:
        Return value of function is dependent on how ``intopt`` is optimized, specifically size of ``group_size_aux``.

        .. code-block::
            mol = pyscf.gto.Mole(atom="O; H 1 0.94; H 1 0.94 2 104.5", basis="def2-TZVP", max_memory=6000).build()
            auxmol = pyscf.df.make_auxmol(mol, "def2-TZVP-ri")
            intopt = gpu4pyscf.df.int3c2e.VHFOpt(mol, auxmol, "int2e")
            intopt.build(diag_block_with_triu=True, aosym=True, group_size_aux=32)
            for idx_k in range(len(intopt.aux_log_qs)):
                j3c_batched = get_int3c2e_by_aux_id(mol, intopt, idx_k)
                print(j3c_batched.shape, j3c_batched.strides)
    """
    nao = mol.nao
    k0, k1 = intopt.aux_ao_loc[idx_k], intopt.aux_ao_loc[idx_k+1]

    if out is None:
        out = cp.zeros([k1 - k0, nao, nao], order="C")

    for idx_ij, _ in enumerate(intopt.log_qs):
        cpi = intopt.cp_idx[idx_ij]
        cpj = intopt.cp_jdx[idx_ij]
        li = intopt.angular[cpi]
        lj = intopt.angular[cpj]
        i0, i1 = intopt.cart_ao_loc[cpi], intopt.cart_ao_loc[cpi+1]
        j0, j1 = intopt.cart_ao_loc[cpj], intopt.cart_ao_loc[cpj+1]

        int3c_slice = cp.zeros([k1-k0, j1-j0, i1-i0], order="C")
        int3c_slice = gpu4pyscf.df.int3c2e.get_int3c2e_slice(intopt, idx_ij, idx_k, out=int3c_slice, omega=omega)
        if not mol.cart:
            int3c_slice = gpu4pyscf.lib.cupy_helper.cart2sph(int3c_slice, axis=1, ang=lj)
            int3c_slice = gpu4pyscf.lib.cupy_helper.cart2sph(int3c_slice, axis=2, ang=li)
        i0, i1 = intopt.ao_loc[cpi], intopt.ao_loc[cpi+1]
        j0, j1 = intopt.ao_loc[cpj], intopt.ao_loc[cpj+1]
        out[:, j0:j1, i0:i1] = int3c_slice
    row, col = np.tril_indices(nao)
    out[:, row, col] = out[:, col, row]

    return out


def _get_j3c_ovl_gpu(mol, intopt, occ_coeff, vir_coeff, j3c_ovl, log):
    """ Inner function for generate and transform 3c-2e ERI to MO basis

    Args:
        mol: pyscf.gto.Mole
        intopt: gpu4pyscf.df.int3c2e.VHFOpt
        occ_coeff: list[cp.ndarray]
        vir_coeff: list[cp.ndarray]
        j3c_ovl: list[np.ndarray or cp.ndarray]
        log: gpu4pyscf.lib.logger.Logger
    """
    nset = len(occ_coeff)
    assert len(occ_coeff) == len(vir_coeff) == len(j3c_ovl)
    j3c_on_gpu = isinstance(j3c_ovl[0], cp.ndarray)
    occ_coeff_sorted = [intopt.sort_orbitals(occ_coeff[iset], axis=[0]) for iset in range(nset)]
    vir_coeff_sorted = [intopt.sort_orbitals(vir_coeff[iset], axis=[0]) for iset in range(nset)]

    dtype = j3c_ovl[0].dtype

    for idx_p in range(len(intopt.aux_log_qs)):
        log.debug(f"processing auxiliary part {idx_p}/{range(len(intopt.aux_log_qs))}")
        if not mol.cart:
            p0, p1 = intopt.sph_aux_loc[idx_p], intopt.sph_aux_loc[idx_p + 1]
        else:
            p0, p1 = intopt.cart_aux_loc[idx_p], intopt.cart_aux_loc[idx_p + 1]
        # obtained j3c is (nbatch_aux, nao, nao)
        j3c = get_int3c2e_by_aux_id(mol, intopt, idx_p)
        nbatch_aux, nao, _ = j3c.shape
        for iset in range(nset):
            co = cp.asarray(occ_coeff_sorted[iset])
            cv = cp.asarray(vir_coeff_sorted[iset])
            nocc, nvir = co.shape[1], cv.shape[1]
            # Puv, vi -> iPu
            j3c_half = co.T @ j3c.reshape(nbatch_aux * nao, nao).T
            j3c_half.shape = (nocc, nbatch_aux, nao)
            # iPu, ua -> iaP
            if j3c_on_gpu:
                for i in range(nocc):
                    j3c_ovl[iset][i, :, p0:p1] = cv.T @ j3c_half[i].T
            else:
                for i in range(nocc):
                    j3c_ovl[iset][i, :, p0:p1] = (cv.T @ j3c_half[i].T).astype(dtype).get(blocking=False)
            co = cv = j3c_half = None
        j3c = None
    cupy.cuda.get_current_stream().synchronize()


def _decompose_j3c(j2c_decomp, j3c, log):
    """ Inner function for decompose 3c-2e ERI (occ-vir part)

    Args:
        j2c_decomp: dict
        j3c: list[np.ndarray or cp.ndarray]
        log: gpu4pyscf.lib.logger.Logger
    """
    j3c_on_gpu = isinstance(j3c[0], cp.ndarray)
    nset = len(j3c)
    naux = j3c[0].shape[2]
    dtype = j3c[0].dtype

    if j3c_on_gpu:
        # directly perform decomposition
        if j2c_decomp["tag"] == "cd":
            j2c_l = cp.asarray(j2c_decomp["j2c_l"], dtype=dtype, order="F")
            # probably memory copy occurs due to c-contiguous array?
            for iset in range(nset):
                shape = j3c[iset].shape
                j3c[iset] = cupyx.scipy.linalg.solve_triangular(
                    j2c_l, j3c[iset].reshape((-1, naux)).T, lower=True, overwrite_b=True).T.reshape(shape)
        elif j2c_decomp["tag"] == "eig":
            j2c_l_inv = cp.asarray(j2c_decomp["j2c_l_inv"], dtype=dtype, order="C")
            for iset in range(nset):
                shape = j3c[iset].shape
                j3c[iset] = (j3c[iset].reshape((-1, naux)) @ j2c_l_inv).reshape(shape)
        else:
            raise ValueError(f"Unknown j2c decomposition tag: {j2c_decomp['tag']}")
    else:
        cp.get_default_memory_pool().free_all_blocks()
        log.debug(f"Available GPU memory: {get_avail_gpu_mem() / 1024**3:.6f} GB")
        fp_avail = 0.7 * get_avail_gpu_mem() / min(j3c[0].strides)
        if j2c_decomp["tag"] == "cd":
            j2c_l = cp.asarray(j2c_decomp["j2c_l"], dtype=dtype, order="F")
            for iset in range(nset):
                shape = j3c[iset].shape
                j3c[iset].shape = (-1, naux)
                n_ov = j3c[iset].shape[0]
                batch_ov = int(fp_avail / (4 * naux))
                log.debug(f"number of batched non-auxiliary indices: {batch_ov}")
                for i_ov in range(0, n_ov, batch_ov):
                    log.debug(f"load non-auxiliary index: {i_ov}/{n_ov}")
                    nbatch_ov = min(batch_ov, n_ov - i_ov)
                    j3c_batched = cp.asarray(j3c[iset][i_ov:i_ov + nbatch_ov])
                    j3c_batched = cupyx.scipy.linalg.solve_triangular(
                        j2c_l, j3c_batched.T, lower=True, overwrite_b=True).T
                    j3c_batched.get(out=j3c[iset][i_ov:i_ov + nbatch_ov], blocking=False)
                    j3c_batched = None
                j3c[iset].shape = shape
        elif j2c_decomp["tag"] == "eig":
            j2c_l_inv = cp.asarray(j2c_decomp["j2c_l_inv"], dtype=dtype, order="C")
            for iset in range(nset):
                shape = j3c[iset].shape
                j3c[iset].shape = (naux, -1)
                n_ov = j3c[iset].shape[1]
                batch_ov = int(fp_avail / (4 * naux))
                for i_ov in range(0, n_ov, batch_ov):
                    nbatch_ov = min(batch_ov, n_ov - i_ov)
                    j3c_batched = cp.asarray(j3c[iset][i_ov:i_ov + nbatch_ov])
                    (j3c_batched @ j2c_l_inv).get(out=j3c[iset][i_ov:i_ov + nbatch_ov], blocking=False)
                    j3c_batched = None
                j3c[iset].shape = shape
        else:
            raise ValueError(f"Unknown j2c decomposition tag: {j2c_decomp['tag']}")
    cupy.cuda.get_current_stream().synchronize()


def get_cderi_ovl_direct_incore_gpu(
        mol, auxmol, occ_coeff, vir_coeff,
        fp_type=CONFIG_FP_TYPE, fp_type_decomp=CONFIG_FP_TYPE_DECOMP, j2c_alg=CONFIG_J2C_ALG,
        cderi_on_gpu=True,
        verbose=None):
    r""" Generate Cholesky decomposed 3c-2e ERI (occ-vir part) in GPU incore directly.

    Note:
        Auxiliary basis sequence is not the same to default ``auxmol``; it is defined in returned value ``intopt``.
        So returned tensor is shuffled (not the same) compared to CPU counterpart.

    Args:
        mol: pyscf.gto.Mole
            Molecule object with normal basis set.

        auxmol: pyscf.gto.Mole
            Molecule object with auxiliary basis set.

        occ_coeff: cp.ndarray or list[cp.ndarray]
            Occupied orbital coefficients. Shape (nset, nao, nocc) or (nao, nocc).

        vir_coeff: cp.ndarray or list[cp.ndarray]
            Virtual orbital coefficients. Shape (nset, nao, nvir) or (nao, nvir).

        fp_type: str
            Floating point type for final returned cderi_ovl (step 4).

            - FP64: Double precision
            - FP32: Single precision

        fp_type_decomp: str
            Floating point for decompose 3c-2e ERI (step 3).

            - FP64: Double precision
            - FP32: Single precision

        cderi_on_gpu: bool
            Whether store cderi on GPU.

        j2c_alg: str
            Algorithm for decomposition.
            - "cd": Cholesky decomposition by default, eigen decomposition when scipy raises error
            - "eig": Eigen decomposition

        verbose: int or None

    Returns:
        dict

        intopt: gpu4pyscf.df.int3c2e.VHFOpt
            Integral optimizer object for 3c-2e ERI on GPU.

        cderi_ovl: cp.ndarray or list[cp.ndarray]
            Cholesky decomposed 3c-2e ERI :math:`Y_{P, i a}` in C-contiguous.
            ``ovl`` refers to (occ, vir, aux (cholesky)).

        j2c_decomp: dict
            Dictionary contains j2c decomposition algorithm and results.
    """
    log = gpu4pyscf.lib.logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    # sanity check and options update
    # only one set of coefficients
    unsqueeze_nset = False
    if occ_coeff.ndim == 2:
        assert vir_coeff.ndim == 2
        occ_coeff = [occ_coeff]
        vir_coeff = [vir_coeff]
        unsqueeze_nset = True
    nao = mol.nao
    nset = len(occ_coeff)
    assert len(vir_coeff) == nset
    nocc_list = []
    nvir_list = []
    for iset in range(nset):
        assert occ_coeff[iset].ndim == 2
        assert occ_coeff[iset].shape[0] == mol.nao
        assert vir_coeff[iset].ndim == 2
        assert vir_coeff[iset].shape[0] == mol.nao
        nocc_list.append(occ_coeff[iset].shape[1])
        nvir_list.append(vir_coeff[iset].shape[1])
    naux = auxmol.nao

    # memory requirement
    assert fp_type.upper() in ["FP64", "FP32"]

    fp_type_decomp = fp_type if fp_type_decomp is None else fp_type_decomp

    # this array will be both scratch and output, so naming is a bit tricky
    # also note that before decomposition, dtype must be FP64
    if cderi_on_gpu:
        cderi_ovl = [cp.empty([nocc, nvir, naux], dtype=get_dtype(fp_type_decomp, True))
                     for (nocc, nvir) in zip(nocc_list, nvir_list)]
    else:
        cderi_ovl = [np.empty([nocc, nvir, naux], dtype=get_dtype(fp_type_decomp, False))
                     for (nocc, nvir) in zip(nocc_list, nvir_list)]

    t1 = log.init_timer()

    # === step 0: generate intopt object ===
    # this object need to be generated before j2c, due to orbital rearrangement
    # except for output, j3c in AO evaluation and AO2MO must be evaluated by FP64 (8 Bytes)
    cp.get_default_memory_pool().free_all_blocks()
    log.debug(f"Available GPU memory: {get_avail_gpu_mem() / 1024**3:.6f} GB")
    fp_required = 2 * naux * naux + nset * nao * nao  # j2c, orbital coefficients
    fp_avail = 0.7 * get_avail_gpu_mem() / 8 - fp_required
    batch_aux = int(fp_avail / (3 * nao * nao))
    log.debug(f"number of batched auxiliary orbitals: {batch_aux}")
    if batch_aux <= MIN_BATCH_AUX_GPU:
        log.warn(f"Auxiliary batch {batch_aux} number too small. Try set to {MIN_BATCH_AUX_GPU} anyway.")
        batch_aux = MIN_BATCH_AUX_GPU
    intopt = gpu4pyscf.df.int3c2e.VHFOpt(mol, auxmol, 'int2e')
    intopt.build(CUTOFF_J3C, diag_block_with_triu=True, aosym=True, group_size=BLKSIZE_AO, group_size_aux=batch_aux)

    # === step 0: generate 2c-2e ERI and decomposition ===
    # We see numerical catastrophe for FP32 in some cases when decomposing j2c by FP32, especially highly polarized or diffusion basis sets.
    # We found that if j2c is decomposed by FP64, this problem may be alleviated, even we may use (already) decomposed j2c in FP32 for generating cderi by triangular solve.
    # In most cases j2c and its decomposition is not computational bottleneck, so we prefer to use high-precision FP64 to do this task without introducing much numerical errors.
    j2c = pyscf.df.incore.fill_2c2e(mol, auxmol)
    j2c = intopt.sort_orbitals(j2c, aux_axis=[0, 1])
    j2c = cp.asarray(j2c, order="C")
    j2c_decomp = get_j2c_decomp_gpu(mol, j2c, alg=j2c_alg, verbose=verbose)
    j2c = None
    t1 = log.timer("generate 2c-2e ERI and decomposition", *t1)

    # === step 1: generate 3c-2e ERI ===
    # === step 2: transform 3c-2e ERI to MO basis ===
    _get_j3c_ovl_gpu(mol, intopt, occ_coeff, vir_coeff, cderi_ovl, log)
    t1 = log.timer("generate and transform 3c-2e ERI", *t1)

    # === step 3: decompose 3c-2e ERI ===
    # transform dtype if necessary
    _decompose_j3c(j2c_decomp, cderi_ovl, log)
    log.timer("decompose 3c-2e ERI", *t1)

    # === step 4: return cderi_ovl ===
    # correctify shape
    for iset in range(nset):
        cderi_ovl[iset].shape = (nocc_list[iset], nvir_list[iset], naux)
    if fp_type_decomp != fp_type:
        for iset in range(nset):
            cderi_ovl[iset] = cderi_ovl[iset].astype(get_dtype(fp_type, cderi_on_gpu))
    cp.cuda.get_current_stream().synchronize()

    log.timer("generate cderi_ovl in GPU incore", *t0)
    if unsqueeze_nset:
        cderi_ovl = cderi_ovl[0]
    return {
        "intopt": intopt,
        "cderi_ovl": cderi_ovl,
        "j2c_decomp": j2c_decomp,
    }

# endregion
