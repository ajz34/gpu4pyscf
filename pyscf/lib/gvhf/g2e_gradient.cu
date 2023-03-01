/*
 * gpu4pyscf is a plugin to use Nvidia GPU in PySCF package
 *
 * Copyright (C) 2022 Qiming Sun
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "gint/g2e.h"
#include "gint/cint2e.cuh"
#include "gint/gout2e.cuh"
#include "gint/rys_roots.cuh"
#include "contract_jk.cuh"

#define POLYFIT_ORDER   5
#define SQRTPIE4        .8862269254527580136
#define PIE4            .7853981633974483096

template<int NROOTS>
__device__
void GINTg0_2e_2d4d(double * __restrict__ g, double * __restrict__ uw,
                    double norm,
                    int ish, int jsh, int ksh, int lsh, int prim_ij,
                    int prim_kl) {
  double * __restrict__ a12 = c_bpcache.a12;
  double * __restrict__ e12 = c_bpcache.e12;
  double * __restrict__ x12 = c_bpcache.x12;
  double * __restrict__ y12 = c_bpcache.y12;
  double * __restrict__ z12 = c_bpcache.z12;
  double aij = a12[prim_ij];
  double akl = a12[prim_kl];
  double eij = e12[prim_ij];
  double ekl = e12[prim_kl];
  double aijkl = aij + akl;
  double a1 = aij * akl;
  double a0 = a1 / aijkl;
  double fac = eij * ekl / (sqrt(aijkl) * a1);

  double * __restrict__ u = uw;
  double * __restrict__ w = u + NROOTS;
  double * __restrict__ gx = g;
  double * __restrict__ gy = g + c_envs.g_size;
  double * __restrict__ gz = g + c_envs.g_size * 2;

  double xij = x12[prim_ij];
  double yij = y12[prim_ij];
  double zij = z12[prim_ij];
  double xkl = x12[prim_kl];
  double ykl = y12[prim_kl];
  double zkl = z12[prim_kl];
  double xijxkl = xij - xkl;
  double yijykl = yij - ykl;
  double zijzkl = zij - zkl;
  int nbas = c_bpcache.nbas;
  double * __restrict__ bas_x = c_bpcache.bas_coords;
  double * __restrict__ bas_y = bas_x + nbas;
  double * __restrict__ bas_z = bas_y + nbas;
  double xixj, yiyj, zizj, xkxl, ykyl, zkzl;
  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];
  double xk = bas_x[ksh];
  double yk = bas_y[ksh];
  double zk = bas_z[ksh];
  double xijxi = xij - xi;
  double yijyi = yij - yi;
  double zijzi = zij - zi;
  double xklxk = xkl - xk;
  double yklyk = ykl - yk;
  double zklzk = zkl - zk;

  int nmax = c_envs.i_l + c_envs.j_l + 1;
  int mmax = c_envs.k_l + c_envs.l_l;
  int ijmin = c_envs.ijmin + 1;
  int klmin = c_envs.klmin;
  int dm = c_envs.stride_klmax;
  int dn = c_envs.stride_ijmax;
  int di = c_envs.stride_ijmax;
  int dj = c_envs.stride_ijmin;
  int dk = c_envs.stride_klmax;
  int dl = c_envs.stride_klmin;
  int dij = c_envs.g_size_ij;
  int i, k;
  int j, l, m, n, off;
  double tmpb0;
  double s0x, s1x, s2x, t0x, t1x;
  double s0y, s1y, s2y, t0y, t1y;
  double s0z, s1z, s2z, t0z, t1z;
  double u2, tmp1, tmp2, tmp3, tmp4;
  double b00, b10, b01, c00x, c00y, c00z, c0px, c0py, c0pz;

  for (i = 0; i < NROOTS; ++i) {
    gx[i] = norm;
    gy[i] = fac;
    gz[i] = w[i];

    u2 = a0 * u[i];
    tmp4 = .5 / (u2 * aijkl + a1);
    b00 = u2 * tmp4;
    tmp1 = 2 * b00;
    tmp2 = tmp1 * akl;
    b10 = b00 + tmp4 * akl;
    c00x = xijxi - tmp2 * xijxkl;
    c00y = yijyi - tmp2 * yijykl;
    c00z = zijzi - tmp2 * zijzkl;

    if (nmax > 0) {
      // gx(irys,0,1) = c00(irys) * gx(irys,0,0)
      // gx(irys,0,n+1) = c00(irys)*gx(irys,0,n) + n*b10(irys)*gx(irys,0,n-1)
      //for (n = 1; n < nmax; ++n) {
      //    off = n * dn;
      //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
      //        gx[j+dn] = c00x[i] * gx[j] + n * b10[i] * gx[j-dn];
      //        gy[j+dn] = c00y[i] * gy[j] + n * b10[i] * gy[j-dn];
      //        gz[j+dn] = c00z[i] * gz[j] + n * b10[i] * gz[j-dn];
      //    }
      //}
      s0x = gx[i];
      s0y = gy[i];
      s0z = gz[i];
      s1x = c00x * s0x;
      s1y = c00y * s0y;
      s1z = c00z * s0z;
      gx[i + dn] = s1x;
      gy[i + dn] = s1y;
      gz[i + dn] = s1z;
      for (n = 1; n < nmax; ++n) {
        s2x = c00x * s1x + n * b10 * s0x;
        s2y = c00y * s1y + n * b10 * s0y;
        s2z = c00z * s1z + n * b10 * s0z;
        gx[i + (n + 1) * dn] = s2x;
        gy[i + (n + 1) * dn] = s2y;
        gz[i + (n + 1) * dn] = s2z;
        s0x = s1x;
        s0y = s1y;
        s0z = s1z;
        s1x = s2x;
        s1y = s2y;
        s1z = s2z;
      }
    }

    if (mmax > 0) {
      // gx(irys,1,0) = c0p(irys) * gx(irys,0,0)
      // gx(irys,m+1,0) = c0p(irys)*gx(irys,m,0) + m*b01(irys)*gx(irys,m-1,0)
      //for (m = 1; m < mmax; ++m) {
      //    off = m * dm;
      //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
      //        gx[j+dm] = c0px[i] * gx[j] + m * b01[i] * gx[j-dm];
      //        gy[j+dm] = c0py[i] * gy[j] + m * b01[i] * gy[j-dm];
      //        gz[j+dm] = c0pz[i] * gz[j] + m * b01[i] * gz[j-dm];
      //    }
      //}
      tmp3 = tmp1 * aij;
      b01 = b00 + tmp4 * aij;
      c0px = xklxk + tmp3 * xijxkl;
      c0py = yklyk + tmp3 * yijykl;
      c0pz = zklzk + tmp3 * zijzkl;
      s0x = gx[i];
      s0y = gy[i];
      s0z = gz[i];
      s1x = c0px * s0x;
      s1y = c0py * s0y;
      s1z = c0pz * s0z;
      gx[i + dm] = s1x;
      gy[i + dm] = s1y;
      gz[i + dm] = s1z;
      for (m = 1; m < mmax; ++m) {
        s2x = c0px * s1x + m * b01 * s0x;
        s2y = c0py * s1y + m * b01 * s0y;
        s2z = c0pz * s1z + m * b01 * s0z;
        gx[i + (m + 1) * dm] = s2x;
        gy[i + (m + 1) * dm] = s2y;
        gz[i + (m + 1) * dm] = s2z;
        s0x = s1x;
        s0y = s1y;
        s0z = s1z;
        s1x = s2x;
        s1y = s2y;
        s1z = s2z;
      }

      if (nmax > 0) {
        // gx(irys,1,1) = c0p(irys)*gx(irys,0,1) + b00(irys)*gx(irys,0,0)
        // gx(irys,m+1,1) = c0p(irys)*gx(irys,m,1)
        // + m*b01(irys)*gx(irys,m-1,1)
        // + b00(irys)*gx(irys,m,0)
        //for (m = 1; m < mmax; ++m) {
        //    off = m * dm + dn;
        //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
        //        gx[j+dm] = c0px[i]*gx[j] + m*b01[i]*gx[j-dm] + b00[i]*gx[j-dn];
        //        gy[j+dm] = c0py[i]*gy[j] + m*b01[i]*gy[j-dm] + b00[i]*gy[j-dn];
        //        gz[j+dm] = c0pz[i]*gz[j] + m*b01[i]*gz[j-dm] + b00[i]*gz[j-dn];
        //    }
        //}
        s0x = gx[i + dn];
        s0y = gy[i + dn];
        s0z = gz[i + dn];
        s1x = c0px * s0x + b00 * gx[i];
        s1y = c0py * s0y + b00 * gy[i];
        s1z = c0pz * s0z + b00 * gz[i];
        gx[i + dn + dm] = s1x;
        gy[i + dn + dm] = s1y;
        gz[i + dn + dm] = s1z;
        for (m = 1; m < mmax; ++m) {
          s2x = c0px * s1x + m * b01 * s0x + b00 * gx[i + m * dm];
          s2y = c0py * s1y + m * b01 * s0y + b00 * gy[i + m * dm];
          s2z = c0pz * s1z + m * b01 * s0z + b00 * gz[i + m * dm];
          gx[i + dn + (m + 1) * dm] = s2x;
          gy[i + dn + (m + 1) * dm] = s2y;
          gz[i + dn + (m + 1) * dm] = s2z;
          s0x = s1x;
          s0y = s1y;
          s0z = s1z;
          s1x = s2x;
          s1y = s2y;
          s1z = s2z;
        }
      }
    }

    // gx(irys,m,n+1) = c00(irys)*gx(irys,m,n)
    // + n*b10(irys)*gx(irys,m,n-1)
    // + m*b00(irys)*gx(irys,m-1,n)
    for (m = 1; m <= mmax; ++m) {
      //for (n = 1; n < nmax; ++n) {
      //    off = m * dm + n * dn;
      //    for (i = 0, j = off; i < NROOTS; ++i, ++j) {
      //        gx[j+dn] = c00x[i]*gx[j] +n*b10[i]*gx[j-dn] + m*b00[i]*gx[j-dm];
      //        gy[j+dn] = c00y[i]*gy[j] +n*b10[i]*gy[j-dn] + m*b00[i]*gy[j-dm];
      //        gz[j+dn] = c00z[i]*gz[j] +n*b10[i]*gz[j-dn] + m*b00[i]*gz[j-dm];
      //    }
      //}
      off = m * dm;
      j = off + i;
      s0x = gx[j];
      s0y = gy[j];
      s0z = gz[j];
      s1x = gx[j + dn];
      s1y = gy[j + dn];
      s1z = gz[j + dn];
      tmpb0 = m * b00;
      for (n = 1; n < nmax; ++n) {
        s2x = c00x * s1x + n * b10 * s0x + tmpb0 * gx[j + n * dn - dm];
        s2y = c00y * s1y + n * b10 * s0y + tmpb0 * gy[j + n * dn - dm];
        s2z = c00z * s1z + n * b10 * s0z + tmpb0 * gz[j + n * dn - dm];
        gx[j + (n + 1) * dn] = s2x;
        gy[j + (n + 1) * dn] = s2y;
        gz[j + (n + 1) * dn] = s2z;
        s0x = s1x;
        s0y = s1y;
        s0z = s1z;
        s1x = s2x;
        s1y = s2y;
        s1z = s2z;
      }
    }
  }

  if (ijmin > 0) {
    // g(i,j) = rirj * g(i,j-1) +  g(i+1,j-1)
    xixj = xi - bas_x[jsh];
    yiyj = yi - bas_y[jsh];
    zizj = zi - bas_z[jsh];
    //for (k = 0; k <= mmax; ++k) {
    //for (j = 0; j < ijmin; ++j) {
    //for (i = nmax-1-j; i >= 0; i--) {
    //    off = k*dk + j*dj + i*di;
    //    for (n = off; n < off+NROOTS; ++n) {
    //        gx[dj+n] = xixj * gx[n] + gx[di+n];
    //        gy[dj+n] = yiyj * gy[n] + gy[di+n];
    //        gz[dj+n] = zizj * gz[n] + gz[di+n];
    //    }
    //} } }

    // unrolling j
    for (j = 0; j < ijmin - 1; j += 2, nmax -= 2) {
      for (k = 0; k <= mmax; ++k) {
        off = k * dk + j * dj;
        for (n = off; n < off + NROOTS; ++n) {
          s0x = gx[n + nmax * di - di];
          s0y = gy[n + nmax * di - di];
          s0z = gz[n + nmax * di - di];
          t1x = xixj * s0x + gx[n + nmax * di];
          t1y = yiyj * s0y + gy[n + nmax * di];
          t1z = zizj * s0z + gz[n + nmax * di];
          gx[dj + n + nmax * di - di] = t1x;
          gy[dj + n + nmax * di - di] = t1y;
          gz[dj + n + nmax * di - di] = t1z;
          s1x = s0x;
          s1y = s0y;
          s1z = s0z;
          for (i = nmax - 2; i >= 0; i--) {
            s0x = gx[n + i * di];
            s0y = gy[n + i * di];
            s0z = gz[n + i * di];
            t0x = xixj * s0x + s1x;
            t0y = yiyj * s0y + s1y;
            t0z = zizj * s0z + s1z;
            gx[dj + n + i * di] = t0x;
            gy[dj + n + i * di] = t0y;
            gz[dj + n + i * di] = t0z;
            gx[dj + dj + n + i * di] = xixj * t0x + t1x;
            gy[dj + dj + n + i * di] = yiyj * t0y + t1y;
            gz[dj + dj + n + i * di] = zizj * t0z + t1z;
            s1x = s0x;
            s1y = s0y;
            s1z = s0z;
            t1x = t0x;
            t1y = t0y;
            t1z = t0z;
          }
        }
      }
    }

    if (j < ijmin) {
      for (k = 0; k <= mmax; ++k) {
        off = k * dk + j * dj;
        for (n = off; n < off + NROOTS; ++n) {
          s1x = gx[n + nmax * di];
          s1y = gy[n + nmax * di];
          s1z = gz[n + nmax * di];
          for (i = nmax - 1; i >= 0; i--) {
            s0x = gx[n + i * di];
            s0y = gy[n + i * di];
            s0z = gz[n + i * di];
            gx[dj + n + i * di] = xixj * s0x + s1x;
            gy[dj + n + i * di] = yiyj * s0y + s1y;
            gz[dj + n + i * di] = zizj * s0z + s1z;
            s1x = s0x;
            s1y = s0y;
            s1z = s0z;
          }
        }
      }
    }
  }

  if (klmin > 0) {
    // g(...,k,l) = rkrl * g(...,k,l-1) + g(...,k+1,l-1)
    xkxl = xk - bas_x[lsh];
    ykyl = yk - bas_y[lsh];
    zkzl = zk - bas_z[lsh];
    //for (l = 0; l < klmin; ++l) {
    //for (k = mmax-1-l; k >= 0; k--) {
    //    off = l*dl + k*dk;
    //    for (n = off; n < off+dij; ++n) {
    //        gx[dl+n] = xkxl * gx[n] + gx[dk+n];
    //        gy[dl+n] = ykyl * gy[n] + gy[dk+n];
    //        gz[dl+n] = zkzl * gz[n] + gz[dk+n];
    //    }
    //} }

    // unrolling l
    for (l = 0; l < klmin - 1; l += 2, mmax -= 2) {
      off = l * dl;
      for (n = off; n < off + dij; ++n) {
        s0x = gx[n + mmax * dk - dk];
        s0y = gy[n + mmax * dk - dk];
        s0z = gz[n + mmax * dk - dk];
        t1x = xkxl * s0x + gx[n + mmax * dk];
        t1y = ykyl * s0y + gy[n + mmax * dk];
        t1z = zkzl * s0z + gz[n + mmax * dk];
        gx[dl + n + mmax * dk - dk] = t1x;
        gy[dl + n + mmax * dk - dk] = t1y;
        gz[dl + n + mmax * dk - dk] = t1z;
        s1x = s0x;
        s1y = s0y;
        s1z = s0z;
        for (k = mmax - 2; k >= 0; k--) {
          s0x = gx[n + k * dk];
          s0y = gy[n + k * dk];
          s0z = gz[n + k * dk];
          t0x = xkxl * s0x + s1x;
          t0y = ykyl * s0y + s1y;
          t0z = zkzl * s0z + s1z;
          gx[dl + n + k * dk] = t0x;
          gy[dl + n + k * dk] = t0y;
          gz[dl + n + k * dk] = t0z;
          gx[dl + dl + n + k * dk] = xkxl * t0x + t1x;
          gy[dl + dl + n + k * dk] = ykyl * t0y + t1y;
          gz[dl + dl + n + k * dk] = zkzl * t0z + t1z;
          s1x = s0x;
          s1y = s0y;
          s1z = s0z;
          t1x = t0x;
          t1y = t0y;
          t1z = t0z;
        }
      }
    }

    if (l < klmin) {
      off = l * dl;
      for (n = off; n < off + dij; ++n) {
        s1x = gx[n + mmax * dk];
        s1y = gy[n + mmax * dk];
        s1z = gz[n + mmax * dk];
        for (k = mmax - 1; k >= 0; k--) {
          s0x = gx[n + k * dk];
          s0y = gy[n + k * dk];
          s0z = gz[n + k * dk];
          gx[dl + n + k * dk] = xkxl * s0x + s1x;
          gy[dl + n + k * dk] = ykyl * s0y + s1y;
          gz[dl + n + k * dk] = zkzl * s0z + s1z;
          s1x = s0x;
          s1y = s0y;
          s1z = s0z;
        }
      }
    }
  }
}


template<int NROOTS>
__device__
void GINTgout2e_nabla1i_per_function(double * __restrict__ g,
                                     double ai, double aj, int i,
                                     double * s_ix, double * s_iy,
                                     double * s_iz,
                                     double * s_jx, double * s_jy,
                                     double * s_jz) {

  int di = c_envs.stride_ijmax;
  int dj = c_envs.stride_ijmin;

  int nf = c_envs.nf;
  int16_t * idx = c_idx4c;

  if (nf > NFffff) {
    idx = c_envs.idx;
  }

  int16_t * idy = idx + nf;
  int16_t * idz = idx + nf * 2;
  int n, ix, iy, iz,
      ij_index_for_ix, i_index_for_ix, j_index_for_ix,
      ij_index_for_iy, i_index_for_iy, j_index_for_iy,
      ij_index_for_iz, i_index_for_iz, j_index_for_iz;

  ix = idx[i];
  ij_index_for_ix = ix % c_envs.g_size_ij;
  i_index_for_ix = ij_index_for_ix % dj / di;
  j_index_for_ix = ij_index_for_ix / dj;
  iy = idy[i];
  ij_index_for_iy = iy % c_envs.g_size_ij;
  i_index_for_iy = ij_index_for_iy % dj / di;
  j_index_for_iy = ij_index_for_iy / dj;
  iz = idz[i];
  ij_index_for_iz = iz % c_envs.g_size_ij;
  i_index_for_iz = ij_index_for_iz % dj / di;
  j_index_for_iz = ij_index_for_iz / dj;

  *s_ix = 0;
  *s_iy = 0;
  *s_iz = 0;
  *s_jx = 0;
  *s_jy = 0;
  *s_jz = 0;

#pragma unroll
  for (n = 0; n < NROOTS; ++n) {
    *s_ix += -i_index_for_ix *
             g[ix + n - di] * g[iy + n] * g[iz + n]
             + 2.0 * ai * g[ix + n + di] * g[iy + n] * g[iz + n];
    *s_iy += -i_index_for_iy *
             g[ix + n] * g[iy + n - di] * g[iz + n]
             + 2.0 * ai * g[ix + n] * g[iy + n + di] * g[iz + n];
    *s_iz += -i_index_for_iz *
             g[ix + n] * g[iy + n] * g[iz + n - di]
             + 2.0 * ai * g[ix + n] * g[iy + n] * g[iz + n + di];
    *s_jx += -j_index_for_ix *
             g[ix + n - dj] * g[iy + n] * g[iz + n]
             + 2.0 * aj * g[ix + n + dj] * g[iy + n] * g[iz + n];
    *s_jy += -j_index_for_iy *
             g[ix + n] * g[iy + n - dj] * g[iz + n]
             + 2.0 * aj * g[ix + n] * g[iy + n + dj] * g[iz + n];
    *s_jz += -j_index_for_iz *
             g[ix + n] * g[iy + n] * g[iz + n - dj]
             + 2.0 * aj * g[ix + n] * g[iy + n] * g[iz + n + dj];
  }
}

template<int NROOTS>
__device__
void GINTgout2e_nabla1i(double * __restrict__ gout, double * __restrict__ g,
                        double ai, double aj) {
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  int i;

  int nf = c_envs.nf;

  for (i = 0; i < c_envs.nf; i++) {
    GINTgout2e_nabla1i_per_function<NROOTS>(g, ai, aj, i,
                                            &s_ix, &s_iy, &s_iz,
                                            &s_jx, &s_jy, &s_jz);

    gout[i] += s_ix;
    gout[i + nf] += s_iy;
    gout[i + 2 * nf] += s_iz;
    gout[i + 3 * nf] += s_jx;
    gout[i + 4 * nf] += s_jy;
    gout[i + 5 * nf] += s_jz;
  }
}

__device__
void GINTkernel_getjk_nabla1i(JKMatrix jk, double * __restrict__ gout,
                              int ish, int jsh, int ksh, int lsh) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int task_id = ty * THREADSX + tx;
  int * ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  // int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nf = c_envs.nf;
  int nao = jk.nao;
  int nao2 = nao * nao;
  int i, j, k, l, n, i_dm;
  int ip, jp, kp, lp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;
  // enough to hold (g,s) shells
  __shared__ double _buf[3 * THREADS * (GPU_CART_MAX * 2 + 1)];
  int n_dm = jk.n_dm;
  double * vj = jk.vj;
  double * vk = jk.vk;
  double * __restrict__ dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;
  if (vk == NULL) {
    if (vj == NULL) {
      return;
    }

    if (nfij > (GPU_CART_MAX * 2 + 1) / 2) {
      double * __restrict__ buf_ij = gout + 6 * c_envs.nf;
      for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        memset(buf_ij, 0, 6 * nfij * sizeof(double));
        double * __restrict__ pgout = gout;
        for (l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n) {
                s_ix = pgout[n];
                s_iy = pgout[n + nf];
                s_iz = pgout[n + 2 * nf];
                s_jx = pgout[n + 3 * nf];
                s_jy = pgout[n + 4 * nf];
                s_jz = pgout[n + 5 * nf];
                buf_ij[n] += s_ix * d_kl;
                buf_ij[n + nfij] += s_iy * d_kl;
                buf_ij[n + 2 * nfij] += s_iz * d_kl;
                buf_ij[n + 3 * nfij] += s_jx * d_kl;
                buf_ij[n + 4 * nfij] += s_jy * d_kl;
                buf_ij[n + 5 * nfij] += s_jz * d_kl;
              }
            }
            pgout += nfij;
          }
        }
        for (n = 0, j = j0; j < j1; ++j) {
          for (i = i0; i < i1; ++i, ++n) {
            atomicAdd(vj + i + nao * j, buf_ij[n]);
            atomicAdd(vj + i + nao * j + nao2, buf_ij[n + nfij]);
            atomicAdd(vj + i + nao * j + 2 * nao2, buf_ij[n + 2 * nfij]);
            atomicAdd(vj + j + nao * i, buf_ij[n + 3 * nfij]);
            atomicAdd(vj + j + nao * i + nao2, buf_ij[n + 4 * nfij]);
            atomicAdd(vj + j + nao * i + 2 * nao2, buf_ij[n + 5 * nfij]);
          }
        }
        dm += nao2;
        vj += 3 * nao2;
      }

    } else {
      for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        for (ip = 0; ip < 6 * nfij; ++ip) {
          _buf[ip * THREADS + task_id] = 0;
        }
        double * __restrict__ pgout = gout;
        for (l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n) {
                s_ix = pgout[n];
                s_iy = pgout[n + nf];
                s_iz = pgout[n + 2 * nf];
                s_jx = pgout[n + 3 * nf];
                s_jy = pgout[n + 4 * nf];
                s_jz = pgout[n + 5 * nf];

                _buf[n * THREADS + task_id] += s_ix * d_kl;
                _buf[(n + nfij) * THREADS + task_id] += s_iy * d_kl;
                _buf[(n + 2 * nfij) * THREADS + task_id] += s_iz * d_kl;
                _buf[(n + 3 * nfij) * THREADS + task_id] += s_jx * d_kl;
                _buf[(n + 4 * nfij) * THREADS + task_id] += s_jy * d_kl;
                _buf[(n + 5 * nfij) * THREADS + task_id] += s_jz * d_kl;
              }
            }
            pgout += nfij;
          }
        }
        for (n = 0, j = j0; j < j1; ++j) {
          for (i = i0; i < i1; ++i, ++n) {
            atomicAdd(vj + i + nao * j, _buf[n * THREADS + task_id]);
            atomicAdd(vj + i + nao * j + nao2,
                      _buf[(n + nfij) * THREADS + task_id]);
            atomicAdd(vj + i + nao * j + 2 * nao2,
                      _buf[(n + 2 * nfij) * THREADS + task_id]);
            atomicAdd(vj + j + nao * i,
                      _buf[(n + 3 * nfij) * THREADS + task_id]);
            atomicAdd(vj + j + nao * i + nao2,
                      _buf[(n + 4 * nfij) * THREADS + task_id]);
            atomicAdd(vj + j + nao * i + 2 * nao2,
                      _buf[(n + 5 * nfij) * THREADS + task_id]);
          }
        }
        dm += nao2;
        vj += 3 * nao2;
      }
    }
    return;
  }

  // vk != NULL
  double * __restrict__ buf_i = _buf;
  double * __restrict__ buf_j = _buf + 3 * nfi * THREADS;

  if (vj != NULL) {
    if (nfij > 10) {
      double * __restrict__ buf_ij = gout + 6 * c_envs.nf;
      for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        memset(buf_ij, 0, 6 * nfij * sizeof(double));
        double * __restrict__ pgout = gout;
        for (l = l0; l < l1; ++l) {
          for (ip = 0; ip < 3 * nfi; ++ip) {
            buf_i[ip * THREADS + task_id] = 0;
          }
          for (jp = 0; jp < 3 * nfj; ++jp) {
            buf_j[jp * THREADS + task_id] = 0;
          }

          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];

            for (n = 0, j = j0; j < j1; ++j) {
              jp = j - j0;
              v_jl_x = 0;
              v_jl_y = 0;
              v_jl_z = 0;
              d_jk = dm[j + nao * k];
              for (i = i0; i < i1; ++i, ++n) {
                ip = i - i0;
                s_ix = pgout[n];
                s_iy = pgout[n + nf];
                s_iz = pgout[n + 2 * nf];
                s_jx = pgout[n + 3 * nf];
                s_jy = pgout[n + 4 * nf];
                s_jz = pgout[n + 5 * nf];
                d_ik = dm[i + nao * k];
                v_jl_x += s_jx * d_ik;
                v_jl_y += s_jy * d_ik;
                v_jl_z += s_jz * d_ik;
                buf_ij[n] += s_ix * d_kl;
                buf_ij[n + nfij] += s_iy * d_kl;
                buf_ij[n + 2 * nfij] += s_iz * d_kl;
                buf_ij[n + 3 * nfij] += s_jx * d_kl;
                buf_ij[n + 4 * nfij] += s_jy * d_kl;
                buf_ij[n + 5 * nfij] += s_jz * d_kl;

                buf_i[ip * THREADS + task_id] += s_ix * d_jk;
                buf_i[(ip + nfi) * THREADS + task_id] += s_iy * d_jk;
                buf_i[(ip + 2 * nfi) * THREADS + task_id] += s_iz * d_jk;
              }
              buf_j[jp * THREADS + task_id] += v_jl_x;
              buf_j[(jp + nfj) * THREADS + task_id] += v_jl_y;
              buf_j[(jp + 2 * nfj) * THREADS + task_id] += v_jl_z;
            }
            pgout += nfij;
          }
          for (ip = 0; ip < nfi; ++ip) {
            atomicAdd(vk + i0 + ip + nao * l, buf_i[ip * THREADS + task_id]);
            atomicAdd(vk + i0 + ip + nao * l + nao2,
                      buf_i[(ip + nfi) * THREADS + task_id]);
            atomicAdd(vk + i0 + ip + nao * l + 2 * nao2,
                      buf_i[(ip + 2 * nfi) * THREADS + task_id]);
          }
          for (jp = 0; jp < nfj; ++jp) {
            atomicAdd(vk + j0 + jp + nao * l, buf_j[jp * THREADS + task_id]);
            atomicAdd(vk + j0 + jp + nao * l + nao2,
                      buf_j[(jp + nfj) * THREADS + task_id]);
            atomicAdd(vk + j0 + jp + nao * l + 2 * nao2,
                      buf_j[(jp + 2 * nfj) * THREADS + task_id]);
          }
        }
        for (n = 0, j = j0; j < j1; ++j) {
          for (i = i0; i < i1; ++i, ++n) {
            atomicAdd(vj + i + nao * j, buf_ij[n]);
            atomicAdd(vj + i + nao * j + nao2, buf_ij[n + nfij]);
            atomicAdd(vj + i + nao * j + 2 * nao2, buf_ij[n + 2 * nfij]);
            atomicAdd(vj + j + nao * i, buf_ij[n + 3 * nfij]);
            atomicAdd(vj + j + nao * i + nao2, buf_ij[n + 4 * nfij]);
            atomicAdd(vj + j + nao * i + 2 * nao2, buf_ij[n + 5 * nfij]);
          }
        }
        dm += nao2;
        vj += 3 * nao2;
        vk += 3 * nao2;
      }

    } else {  // nfij <= s * f
      double * __restrict__ buf_ij = buf_j + 3 * nfj * THREADS;
      for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        for (ip = 0; ip < 6 * nfij; ++ip) {
          buf_ij[ip * THREADS + task_id] = 0;
        }
        double * __restrict__ pgout = gout;
        for (l = l0; l < l1; ++l) {
          for (ip = 0; ip < 3 * nfi; ++ip) {
            buf_i[ip * THREADS + task_id] = 0;
          }
          for (jp = 0; jp < 3 * nfj; ++jp) {
            buf_j[jp * THREADS + task_id] = 0;
          }

          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              jp = j - j0;
              v_jl_x = 0;
              v_jl_y = 0;
              v_jl_z = 0;
              d_jk = dm[j + nao * k];
              for (i = i0; i < i1; ++i, ++n) {
                ip = i - i0;
                s_ix = pgout[n];
                s_iy = pgout[n + nf];
                s_iz = pgout[n + 2 * nf];
                s_jx = pgout[n + 3 * nf];
                s_jy = pgout[n + 4 * nf];
                s_jz = pgout[n + 5 * nf];

                d_ik = dm[i + nao * k];
                v_jl_x += s_jx * d_ik;
                v_jl_y += s_jy * d_ik;
                v_jl_z += s_jz * d_ik;

                buf_ij[n * THREADS + task_id] += s_ix * d_kl;
                buf_ij[(n + nfij) * THREADS + task_id] += s_iy * d_kl;
                buf_ij[(n + 2 * nfij) * THREADS + task_id] += s_iz * d_kl;
                buf_ij[(n + 3 * nfij) * THREADS + task_id] += s_jx * d_kl;
                buf_ij[(n + 4 * nfij) * THREADS + task_id] += s_jy * d_kl;
                buf_ij[(n + 5 * nfij) * THREADS + task_id] += s_jz * d_kl;

                buf_i[ip * THREADS + task_id] += s_ix * d_jk;
                buf_i[(ip + nfi) * THREADS + task_id] += s_iy * d_jk;
                buf_i[(ip + 2 * nfi) * THREADS + task_id] += s_iz * d_jk;
              }
              buf_j[jp * THREADS + task_id] += v_jl_x;
              buf_j[(jp + nfj) * THREADS + task_id] += v_jl_y;
              buf_j[(jp + 2 * nfj) * THREADS + task_id] += v_jl_z;
            }
            pgout += nfij;
          }
          for (ip = 0; ip < nfi; ++ip) {
            atomicAdd(vk + i0 + ip + nao * l, buf_i[ip * THREADS + task_id]);
            atomicAdd(vk + i0 + ip + nao * l + nao2,
                      buf_i[(ip + nfi) * THREADS + task_id]);
            atomicAdd(vk + i0 + ip + nao * l + 2 * nao2,
                      buf_i[(ip + 2 * nfi) * THREADS + task_id]);
          }
          for (jp = 0; jp < nfj; ++jp) {
            atomicAdd(vk + j0 + jp + nao * l, buf_j[jp * THREADS + task_id]);
            atomicAdd(vk + j0 + jp + nao * l + nao2,
                      buf_j[(jp + nfj) * THREADS + task_id]);
            atomicAdd(vk + j0 + jp + nao * l + 2 * nao2,
                      buf_j[(jp + 2 * nfj) * THREADS + task_id]);
          }
        }
        for (n = 0, j = j0; j < j1; ++j) {
          for (i = i0; i < i1; ++i, ++n) {
            atomicAdd(vj + i + nao * j, buf_ij[n * THREADS + task_id]);
            atomicAdd(vj + i + nao * j + nao2,
                      buf_ij[(n + nfij) * THREADS + task_id]);
            atomicAdd(vj + i + nao * j + 2 * nao2,
                      buf_ij[(n + 2 * nfij) * THREADS + task_id]);
            atomicAdd(vj + j + nao * i,
                      buf_ij[(n + 3 * nfij) * THREADS + task_id]);
            atomicAdd(vj + j + nao * i + nao2,
                      buf_ij[(n + 4 * nfij) * THREADS + task_id]);
            atomicAdd(vj + j + nao * i + 2 * nao2,
                      buf_ij[(n + 5 * nfij) * THREADS + task_id]);
          }
        }
        dm += nao2;
        vj += 3 * nao2;
        vk += 3 * nao2;
      }
    }

  } else {  // vj == NULL, vk != NULL
    for (i_dm = 0; i_dm < n_dm; ++i_dm) {
      for (n = 0, l = l0; l < l1; ++l) {
        for (ip = 0; ip < 3 * nfi; ++ip) {
          buf_i[ip * THREADS + task_id] = 0;
        }
        for (jp = 0; jp < 3 * nfj; ++jp) {
          buf_j[jp * THREADS + task_id] = 0;
        }

        for (k = k0; k < k1; ++k) {
          for (j = j0; j < j1; ++j) {
            jp = j - j0;
            v_jl_x = 0;
            v_jl_y = 0;
            v_jl_z = 0;
            d_jk = dm[j + nao * k];
            for (i = i0; i < i1; ++i, ++n) {
              ip = i - i0;
              s_ix = gout[n];
              s_iy = gout[n + nf];
              s_iz = gout[n + 2 * nf];
              s_jx = gout[n + 3 * nf];
              s_jy = gout[n + 4 * nf];
              s_jz = gout[n + 5 * nf];
              d_ik = dm[i + nao * k];
              v_jl_x += s_jx * d_ik;
              v_jl_y += s_jy * d_ik;
              v_jl_z += s_jz * d_ik;

              buf_i[ip * THREADS + task_id] += s_ix * d_jk;
              buf_i[(ip + nfi) * THREADS + task_id] += s_iy * d_jk;
              buf_i[(ip + 2 * nfi) * THREADS + task_id] += s_iz * d_jk;
            }
            buf_j[jp * THREADS + task_id] += v_jl_x;
            buf_j[(jp + nfj) * THREADS + task_id] += v_jl_y;
            buf_j[(jp + 2 * nfj) * THREADS + task_id] += v_jl_z;
          }
        }
        for (ip = 0; ip < nfi; ++ip) {
          atomicAdd(vk + i0 + ip + nao * l, buf_i[ip * THREADS + task_id]);
          atomicAdd(vk + i0 + ip + nao * l + nao2,
                    buf_i[(ip + nfi) * THREADS + task_id]);
          atomicAdd(vk + i0 + ip + nao * l + 2 * nao2,
                    buf_i[(ip + 2 * nfi) * THREADS + task_id]);
        }
        for (jp = 0; jp < nfj; ++jp) {
          atomicAdd(vk + j0 + jp + nao * l, buf_j[jp * THREADS + task_id]);
          atomicAdd(vk + j0 + jp + nao * l + nao2,
                    buf_j[(jp + nfj) * THREADS + task_id]);
          atomicAdd(vk + j0 + jp + nao * l + 2 * nao2,
                    buf_j[(jp + 2 * nfj) * THREADS + task_id]);
        }
      }
      dm += nao2;
      vk += 3 * nao2;
    }
  }


  // vj == NULL, vk != NULL
  vk = jk.vk;
  dm = jk.dm;
  for (i_dm = 0; i_dm < n_dm; ++i_dm) {
    for (k = k0; k < k1; ++k) {
      kp = k - k0;
      for (ip = 0; ip < 3 * nfi; ++ip) {
        buf_i[ip * THREADS + task_id] = 0;
      }
      for (jp = 0; jp < 3 * nfj; ++jp) {
        buf_j[jp * THREADS + task_id] = 0;
      }

      for (l = l0; l < l1; ++l) {
        lp = l - l0;
        n = nfij * (lp * nfk + kp);
        for (j = j0; j < j1; ++j) {
          jp = j - j0;
          v_jk_x = 0;
          v_jk_y = 0;
          v_jk_z = 0;
          d_jl = dm[j + nao * l];
          for (i = i0; i < i1; ++i, ++n) {
            ip = i - i0;
            s_ix = gout[n];
            s_iy = gout[n + nf];
            s_iz = gout[n + 2 * nf];
            s_jx = gout[n + 3 * nf];
            s_jy = gout[n + 4 * nf];
            s_jz = gout[n + 5 * nf];

            d_il = dm[i + nao * l];
            v_jk_x += s_jx * d_il;
            v_jk_y += s_jy * d_il;
            v_jk_z += s_jz * d_il;

            buf_i[ip * THREADS + task_id] += s_ix * d_jl;
            buf_i[(ip + nfi) * THREADS + task_id] += s_iy * d_jl;
            buf_i[(ip + 2 * nfi) * THREADS + task_id] += s_iz * d_jl;
          }
          buf_j[jp * THREADS + task_id] += v_jk_x;
          buf_j[(jp + nfj) * THREADS + task_id] += v_jk_y;
          buf_j[(jp + 2 * nfj) * THREADS + task_id] += v_jk_z;
        }
      }
      for (ip = 0; ip < nfi; ++ip) {
        atomicAdd(vk + i0 + ip + nao * k, buf_i[ip * THREADS + task_id]);
        atomicAdd(vk + i0 + ip + nao * k + nao2,
                  buf_i[(ip + nfi) * THREADS + task_id]);
        atomicAdd(vk + i0 + ip + nao * k + 2 * nao2,
                  buf_i[(ip + 2 * nfi) * THREADS + task_id]);
      }
      for (jp = 0; jp < nfj; ++jp) {
        atomicAdd(vk + j0 + jp + nao * k, buf_j[jp * THREADS + task_id]);
        atomicAdd(vk + j0 + jp + nao * k + nao2,
                  buf_j[(jp + nfj) * THREADS + task_id]);
        atomicAdd(vk + j0 + jp + nao * k + 2 * nao2,
                  buf_j[(jp + 2 * nfj) * THREADS + task_id]);
      }
    }
    dm += nao * nao;
    vk += 3 * nao * nao;
  }
}

template<int NROOTS, int GOUTSIZE>
__global__
static void GINTint2e_jk_kernel_nabla1i(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int task_id = threadIdx.y * THREADSX + threadIdx.x;

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f, i_dm;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;


  int n_dm = jk.n_dm;
  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * NROOTS *
                2;
  double gout[GOUTSIZE];
  double * __restrict__ g =
      gout + (3 * nfik + 3 * nfjk + 3 * nfil + 3 * nfjl + 6 * nfij) * n_dm;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }
  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    if (nfij > (GPU_CART_MAX * 2 + 1) / 2 / n_dm) {
      double * __restrict__ buf_ij = gout;
      memset(buf_ij, 0, 6 * nfij * n_dm * sizeof(double));


      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<NROOTS>(g, uw, norm,
                                 as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          buf_ij = gout;
          dm = jk.dm;
          for (i_dm = 0; i_dm < n_dm; ++i_dm) {
            for (f = 0, l = l0; l < l1; ++l) {
              for (k = k0; k < k1; ++k) {
                d_kl = dm[k + nao * l];
                for (n = 0, j = j0; j < j1; ++j) {
                  for (i = i0; i < i1; ++i, ++n, ++f) {
                    GINTgout2e_nabla1i_per_function<NROOTS>(g, ai, aj, f,
                                                            &s_ix, &s_iy, &s_iz,
                                                            &s_jx, &s_jy,
                                                            &s_jz);
                    buf_ij[n] += s_ix * d_kl;
                    buf_ij[n + nfij] += s_iy * d_kl;
                    buf_ij[n + 2 * nfij] += s_iz * d_kl;
                    buf_ij[n + 3 * nfij] += s_jx * d_kl;
                    buf_ij[n + 4 * nfij] += s_jy * d_kl;
                    buf_ij[n + 5 * nfij] += s_jz * d_kl;
                  }
                }
              }
            }
            dm += nao2;
            buf_ij += 6 * nfij;
          }
          uw += NROOTS * 2;
        }
      }

      buf_ij = gout;
      for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        for (n = 0, j = j0; j < j1; ++j) {
          for (i = i0; i < i1; ++i, ++n) {
            atomicAdd(vj + i + nao * j, buf_ij[n]);
            atomicAdd(vj + i + nao * j + nao2, buf_ij[n + nfij]);
            atomicAdd(vj + i + nao * j + 2 * nao2, buf_ij[n + 2 * nfij]);
            atomicAdd(vj + j + nao * i, buf_ij[n + 3 * nfij]);
            atomicAdd(vj + j + nao * i + nao2, buf_ij[n + 4 * nfij]);
            atomicAdd(vj + j + nao * i + 2 * nao2, buf_ij[n + 5 * nfij]);
          }
        }
        vj += 3 * nao2;
        buf_ij += 6 * nfij;
      }
    } else {

      extern __shared__ double _buf[];

      for (ip = 0; ip < 6 * nfij * n_dm; ++ip) {
        _buf[ip * THREADS + task_id] = 0;
      }


      double * __restrict__ buf_ij;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<NROOTS>(g, uw, norm,
                                 as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          buf_ij = _buf;
          dm = jk.dm;
          for (i_dm = 0; i_dm < n_dm; ++i_dm) {
            for (f = 0, l = l0; l < l1; ++l) {
              for (k = k0; k < k1; ++k) {
                d_kl = dm[k + nao * l];
                for (n = 0, j = j0; j < j1; ++j) {
                  for (i = i0; i < i1; ++i, ++n, ++f) {
                    GINTgout2e_nabla1i_per_function<NROOTS>(g, ai, aj, f,
                                                            &s_ix, &s_iy, &s_iz,
                                                            &s_jx, &s_jy,
                                                            &s_jz);
                    buf_ij[n * THREADS + task_id] += s_ix * d_kl;
                    buf_ij[(n + nfij) * THREADS + task_id] += s_iy * d_kl;
                    buf_ij[(n + 2 * nfij) * THREADS + task_id] += s_iz * d_kl;
                    buf_ij[(n + 3 * nfij) * THREADS + task_id] += s_jx * d_kl;
                    buf_ij[(n + 4 * nfij) * THREADS + task_id] += s_jy * d_kl;
                    buf_ij[(n + 5 * nfij) * THREADS + task_id] += s_jz * d_kl;
                  }
                }
              }
            }
            dm += nao2;
            buf_ij += 6 * nfij * THREADS;
          }
          uw += NROOTS * 2;
        }
      }

      buf_ij = _buf;
      for (i_dm = 0; i_dm < n_dm; ++i_dm) {
        for (n = 0, j = j0; j < j1; ++j) {
          for (i = i0; i < i1; ++i, ++n) {
            atomicAdd(vj + i + nao * j, buf_ij[n * THREADS + task_id]);
            atomicAdd(vj + i + nao * j + nao2,
                      buf_ij[(n + nfij) * THREADS + task_id]);
            atomicAdd(vj + i + nao * j + 2 * nao2,
                      buf_ij[(n + 2 * nfij) * THREADS + task_id]);
            atomicAdd(vj + j + nao * i,
                      buf_ij[(n + 3 * nfij) * THREADS + task_id]);
            atomicAdd(vj + j + nao * i + nao2,
                      buf_ij[(n + 4 * nfij) * THREADS + task_id]);
            atomicAdd(vj + j + nao * i + 2 * nao2,
                      buf_ij[(n + 5 * nfij) * THREADS + task_id]);
          }
        }
        vj += 3 * nao2;
        buf_ij += 6 * nfij * THREADS;
      }
    }
  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij;
      double * __restrict__ buf_ik = gout;
      double * __restrict__ buf_jk = buf_ik + 3 * nfik * n_dm;
      double * __restrict__ buf_il = buf_jk + 3 * nfjk * n_dm;
      double * __restrict__ buf_jl = buf_il + 3 * nfil * n_dm;
      if (nfij > (GPU_CART_MAX * 2 + 1) / 2 / n_dm) {
        double * __restrict__ buf_ij = buf_jl + 3 * nfjl * n_dm;
        memset(gout, 0,
               (3 * nfik + 3 * nfjk + 3 * nfil + 3 * nfjl + 6 * nfij) * n_dm *
               sizeof(double));

        for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
          double ai = exponent_i[(ij - prim_ij) / nprim_j];
          double aj = exponent_j[(ij - prim_ij) % nprim_j];
          for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
            GINTg0_2e_2d4d<NROOTS>(g, uw, norm,
                                   as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
            dm = jk.dm;
            p_buf_ij = buf_ij;
            for (i_dm = 0; i_dm < n_dm; ++i_dm) {
              p_buf_il = buf_il + 3 * i_dm * nfil;
              p_buf_jl = buf_jl + 3 * i_dm * nfjl;
              for (f = 0, l = l0; l < l1; ++l) {
                p_buf_ik = buf_ik + 3 * i_dm * nfik;
                p_buf_jk = buf_jk + 3 * i_dm * nfjk;
                for (k = k0; k < k1; ++k) {
                  d_kl = dm[k + nao * l];
                  for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                    d_jl = dm[j + nao * l];
                    d_jk = dm[j + nao * k];

                    v_jl_x = 0;
                    v_jl_y = 0;
                    v_jl_z = 0;
                    v_jk_x = 0;
                    v_jk_y = 0;
                    v_jk_z = 0;

                    for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                      d_il = dm[i + nao * l];
                      d_ik = dm[i + nao * k];

                      GINTgout2e_nabla1i_per_function<NROOTS>(g, ai, aj, f,
                                                              &s_ix, &s_iy,
                                                              &s_iz,
                                                              &s_jx, &s_jy,
                                                              &s_jz);
                      p_buf_ij[n] += s_ix * d_kl;
                      p_buf_ij[n + nfij] += s_iy * d_kl;
                      p_buf_ij[n + 2 * nfij] += s_iz * d_kl;
                      p_buf_ij[n + 3 * nfij] += s_jx * d_kl;
                      p_buf_ij[n + 4 * nfij] += s_jy * d_kl;
                      p_buf_ij[n + 5 * nfij] += s_jz * d_kl;

                      p_buf_ik[ip] += s_ix * d_jl;
                      p_buf_ik[ip + nfik] += s_iy * d_jl;
                      p_buf_ik[ip + 2 * nfik] += s_iz * d_jl;

                      p_buf_il[ip] += s_ix * d_jk;
                      p_buf_il[ip + nfil] += s_iy * d_jk;
                      p_buf_il[ip + 2 * nfil] += s_iz * d_jk;

                      v_jl_x += s_jx * d_ik;
                      v_jl_y += s_jy * d_ik;
                      v_jl_z += s_jz * d_ik;

                      v_jk_x += s_jx * d_il;
                      v_jk_y += s_jy * d_il;
                      v_jk_z += s_jz * d_il;
                    }

                    p_buf_jl[jp] += v_jl_x;
                    p_buf_jl[jp + nfjl] += v_jl_y;
                    p_buf_jl[jp + 2 * nfjl] += v_jl_z;

                    p_buf_jk[jp] += v_jk_x;
                    p_buf_jk[jp + nfjk] += v_jk_y;
                    p_buf_jk[jp + 2 * nfjk] += v_jk_z;
                  }

                  p_buf_jk += nfj;
                  p_buf_ik += nfi;
                }

                p_buf_il += nfi;
                p_buf_jl += nfj;
              }
              dm += nao2;
              p_buf_ij += 6 * nfij;
            }
            uw += NROOTS * 2;
          }
        }

        p_buf_il = buf_il;
        p_buf_jl = buf_jl;
        p_buf_ik = buf_ik;
        p_buf_jk = buf_jk;
        p_buf_ij = buf_ij;
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
          for (n = 0, j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
              atomicAdd(vj + i + nao * j, p_buf_ij[n]);
              atomicAdd(vj + i + nao * j + nao2, p_buf_ij[n + nfij]);
              atomicAdd(vj + i + nao * j + 2 * nao2, p_buf_ij[n + 2 * nfij]);
              atomicAdd(vj + j + nao * i, p_buf_ij[n + 3 * nfij]);
              atomicAdd(vj + j + nao * i + nao2, p_buf_ij[n + 4 * nfij]);
              atomicAdd(vj + j + nao * i + 2 * nao2, p_buf_ij[n + 5 * nfij]);
            }
          }

          for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
            for (i = i0; i < i1; ++i, ++ip) {
              atomicAdd(vk + i + nao * k, p_buf_ik[ip]);
              atomicAdd(vk + i + nao * k + nao2, p_buf_ik[ip + nfik]);
              atomicAdd(vk + i + nao * k + 2 * nao2, p_buf_ik[ip + 2 * nfik]);
            }

            for (j = j0; j < j1; ++j, ++jp) {
              atomicAdd(vk + j + nao * k, p_buf_jk[jp]);
              atomicAdd(vk + j + nao * k + nao2, p_buf_jk[jp + nfjk]);
              atomicAdd(vk + j + nao * k + 2 * nao2, p_buf_jk[jp + 2 * nfjk]);
            }
          }

          for (ip = 0, jp = 0, n = 0, l = l0; l < l1; ++l) {
            for (i = i0; i < i1; ++i, ++ip) {
              atomicAdd(vk + i + nao * l, p_buf_il[ip]);
              atomicAdd(vk + i + nao * l + nao2, p_buf_il[ip + nfil]);
              atomicAdd(vk + i + nao * l + 2 * nao2, p_buf_il[ip + 2 * nfil]);
            }

            for (j = j0; j < j1; ++j, ++jp) {
              atomicAdd(vk + j + nao * l, p_buf_jl[jp]);
              atomicAdd(vk + j + nao * l + nao2, p_buf_jl[jp + nfjl]);
              atomicAdd(vk + j + nao * l + 2 * nao2, p_buf_jl[jp + 2 * nfjl]);
            }
          }

          vj += 3 * nao2;
          vk += 3 * nao2;
          p_buf_il += 3 * nfil;
          p_buf_jl += 3 * nfjl;
          p_buf_ik += 3 * nfik;
          p_buf_jk += 3 * nfjk;
          p_buf_ij += 6 * nfij;
        }
      } else {
        extern __shared__ double buf_ij[];

        memset(gout, 0,
               (3 * nfik + 3 * nfjk + 3 * nfil + 3 * nfjl) * n_dm *
               sizeof(double));

        double * __restrict__ p_buf_ij;

        for (ip = 0; ip < 6 * nfij * n_dm; ++ip) {
          buf_ij[ip * THREADS + task_id] = 0;
        }

        for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
          double ai = exponent_i[(ij - prim_ij) / nprim_j];
          double aj = exponent_j[(ij - prim_ij) % nprim_j];
          for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
            GINTg0_2e_2d4d<NROOTS>(g, uw, norm,
                                   as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
            p_buf_ij = buf_ij;
            dm = jk.dm;
            for (i_dm = 0; i_dm < n_dm; ++i_dm) {
              p_buf_il = buf_il + 3 * i_dm * nfil;
              p_buf_jl = buf_jl + 3 * i_dm * nfjl;
              for (f = 0, l = l0; l < l1; ++l) {
                p_buf_ik = buf_ik + 3 * i_dm * nfik;
                p_buf_jk = buf_jk + 3 * i_dm * nfjk;
                for (k = k0; k < k1; ++k) {
                  d_kl = dm[k + nao * l];
                  for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                    d_jl = dm[j + nao * l];
                    d_jk = dm[j + nao * k];

                    v_jl_x = 0;
                    v_jl_y = 0;
                    v_jl_z = 0;
                    v_jk_x = 0;
                    v_jk_y = 0;
                    v_jk_z = 0;

                    for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                      d_il = dm[i + nao * l];
                      d_ik = dm[i + nao * k];

                      GINTgout2e_nabla1i_per_function<NROOTS>(g, ai, aj, f,
                                                              &s_ix, &s_iy,
                                                              &s_iz,
                                                              &s_jx, &s_jy,
                                                              &s_jz);
                      p_buf_ij[n * THREADS + task_id] += s_ix * d_kl;
                      p_buf_ij[(n + nfij) * THREADS + task_id] += s_iy * d_kl;
                      p_buf_ij[(n + 2 * nfij) * THREADS + task_id] +=
                          s_iz * d_kl;
                      p_buf_ij[(n + 3 * nfij) * THREADS + task_id] +=
                          s_jx * d_kl;
                      p_buf_ij[(n + 4 * nfij) * THREADS + task_id] +=
                          s_jy * d_kl;
                      p_buf_ij[(n + 5 * nfij) * THREADS + task_id] +=
                          s_jz * d_kl;

                      p_buf_ik[ip] += s_ix * d_jl;
                      p_buf_ik[ip + nfik] += s_iy * d_jl;
                      p_buf_ik[ip + 2 * nfik] += s_iz * d_jl;

                      p_buf_il[ip] += s_ix * d_jk;
                      p_buf_il[ip + nfil] += s_iy * d_jk;
                      p_buf_il[ip + 2 * nfil] += s_iz * d_jk;

                      v_jl_x += s_jx * d_ik;
                      v_jl_y += s_jy * d_ik;
                      v_jl_z += s_jz * d_ik;

                      v_jk_x += s_jx * d_il;
                      v_jk_y += s_jy * d_il;
                      v_jk_z += s_jz * d_il;
                    }

                    p_buf_jl[jp] += v_jl_x;
                    p_buf_jl[jp + nfjl] += v_jl_y;
                    p_buf_jl[jp + 2 * nfjl] += v_jl_z;

                    p_buf_jk[jp] += v_jk_x;
                    p_buf_jk[jp + nfjk] += v_jk_y;
                    p_buf_jk[jp + 2 * nfjk] += v_jk_z;
                  }

                  p_buf_jk += nfj;
                  p_buf_ik += nfi;
                }

                p_buf_il += nfi;
                p_buf_jl += nfj;
              }
              dm += nao2;
              p_buf_ij += 6 * nfij * THREADS;
            }
            uw += NROOTS * 2;
          }
        }

        p_buf_il = buf_il;
        p_buf_jl = buf_jl;
        p_buf_ik = buf_ik;
        p_buf_jk = buf_jk;
        p_buf_ij = buf_ij;
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {
          for (n = 0, j = j0; j < j1; ++j) {
            for (i = i0; i < i1; ++i, ++n) {
              atomicAdd(vj + i + nao * j, p_buf_ij[n * THREADS + task_id]);
              atomicAdd(vj + i + nao * j + nao2, p_buf_ij[(n + nfij) * THREADS + task_id]);
              atomicAdd(vj + i + nao * j + 2 * nao2, p_buf_ij[(n + 2 * nfij) * THREADS + task_id]);
              atomicAdd(vj + j + nao * i, p_buf_ij[(n + 3 * nfij) * THREADS + task_id]);
              atomicAdd(vj + j + nao * i + nao2, p_buf_ij[(n + 4 * nfij) * THREADS + task_id]);
              atomicAdd(vj + j + nao * i + 2 * nao2, p_buf_ij[(n + 5 * nfij) * THREADS + task_id]);
            }
          }

          for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
            for (i = i0; i < i1; ++i, ++ip) {
              atomicAdd(vk + i + nao * k, p_buf_ik[ip]);
              atomicAdd(vk + i + nao * k + nao2, p_buf_ik[ip + nfik]);
              atomicAdd(vk + i + nao * k + 2 * nao2, p_buf_ik[ip + 2 * nfik]);
            }

            for (j = j0; j < j1; ++j, ++jp) {
              atomicAdd(vk + j + nao * k, p_buf_jk[jp]);
              atomicAdd(vk + j + nao * k + nao2, p_buf_jk[jp + nfjk]);
              atomicAdd(vk + j + nao * k + 2 * nao2, p_buf_jk[jp + 2 * nfjk]);
            }
          }

          for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
            for (i = i0; i < i1; ++i, ++ip) {
              atomicAdd(vk + i + nao * l, p_buf_il[ip]);
              atomicAdd(vk + i + nao * l + nao2, p_buf_il[ip + nfil]);
              atomicAdd(vk + i + nao * l + 2 * nao2, p_buf_il[ip + 2 * nfil]);
            }

            for (j = j0; j < j1; ++j, ++jp) {
              atomicAdd(vk + j + nao * l, p_buf_jl[jp]);
              atomicAdd(vk + j + nao * l + nao2, p_buf_jl[jp + nfjl]);
              atomicAdd(vk + j + nao * l + 2 * nao2, p_buf_jl[jp + 2 * nfjl]);
            }
          }

          vj += 3 * nao2;
          vk += 3 * nao2;
          p_buf_il += 3 * nfil;
          p_buf_jl += 3 * nfjl;
          p_buf_ik += 3 * nfik;
          p_buf_jk += 3 * nfjk;
          p_buf_ij += 6 * nfij * THREADS;
        }
      }

    } else { // only vk required
      if (nfik + nfil > (GPU_CART_MAX * 2 + 1) / n_dm) {

        double * __restrict__ buf_ik = gout;
        double * __restrict__ buf_jk = buf_ik + 3 * nfik * n_dm;
        double * __restrict__ buf_il = buf_jk + 3 * nfjk * n_dm;
        double * __restrict__ buf_jl = buf_il + 3 * nfil * n_dm;

        memset(gout, 0,
               (3 * nfik + 3 * nfjk + 3 * nfil + 3 * nfjl) * n_dm *
               sizeof(double));

        for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
          double ai = exponent_i[(ij - prim_ij) / nprim_j];
          double aj = exponent_j[(ij - prim_ij) % nprim_j];
          for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
            GINTg0_2e_2d4d<NROOTS>(g, uw, norm,
                                   as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
            dm = jk.dm;
            for (i_dm = 0; i_dm < n_dm; ++i_dm) {
              p_buf_il = buf_il + 3 * i_dm * nfil;
              p_buf_jl = buf_jl + 3 * i_dm * nfjl;
              for (f = 0, l = l0; l < l1; ++l) {
                p_buf_ik = buf_ik + 3 * i_dm * nfik;
                p_buf_jk = buf_jk + 3 * i_dm * nfjk;
                for (k = k0; k < k1; ++k) {
                  for (jp = 0, j = j0; j < j1; ++j, ++jp) {
                    d_jl = dm[j + nao * l];
                    d_jk = dm[j + nao * k];

                    v_jl_x = 0;
                    v_jl_y = 0;
                    v_jl_z = 0;
                    v_jk_x = 0;
                    v_jk_y = 0;
                    v_jk_z = 0;

                    for (ip = 0, i = i0; i < i1; ++i, ++ip, ++f) {
                      d_il = dm[i + nao * l];
                      d_ik = dm[i + nao * k];

                      GINTgout2e_nabla1i_per_function<NROOTS>(g, ai, aj, f,
                                                              &s_ix, &s_iy,
                                                              &s_iz,
                                                              &s_jx, &s_jy,
                                                              &s_jz);

                      p_buf_ik[ip] += s_ix * d_jl;
                      p_buf_ik[ip + nfik] += s_iy * d_jl;
                      p_buf_ik[ip + 2 * nfik] += s_iz * d_jl;

                      p_buf_il[ip] += s_ix * d_jk;
                      p_buf_il[ip + nfil] += s_iy * d_jk;
                      p_buf_il[ip + 2 * nfil] += s_iz * d_jk;

                      v_jl_x += s_jx * d_ik;
                      v_jl_y += s_jy * d_ik;
                      v_jl_z += s_jz * d_ik;

                      v_jk_x += s_jx * d_il;
                      v_jk_y += s_jy * d_il;
                      v_jk_z += s_jz * d_il;
                    }

                    p_buf_jl[jp] += v_jl_x;
                    p_buf_jl[jp + nfjl] += v_jl_y;
                    p_buf_jl[jp + 2 * nfjl] += v_jl_z;

                    p_buf_jk[jp] += v_jk_x;
                    p_buf_jk[jp + nfjk] += v_jk_y;
                    p_buf_jk[jp + 2 * nfjk] += v_jk_z;
                  }

                  p_buf_jk += nfj;
                  p_buf_ik += nfi;
                }

                p_buf_il += nfi;
                p_buf_jl += nfj;
              }
              dm += nao2;
            }
            uw += NROOTS * 2;
          }
        }

        p_buf_il = buf_il;
        p_buf_jl = buf_jl;
        p_buf_ik = buf_ik;
        p_buf_jk = buf_jk;
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {

          for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
            for (i = i0; i < i1; ++i, ++ip) {
              atomicAdd(vk + i + nao * k, p_buf_ik[ip]);
              atomicAdd(vk + i + nao * k + nao2, p_buf_ik[ip + nfik]);
              atomicAdd(vk + i + nao * k + 2 * nao2, p_buf_ik[ip + 2 * nfik]);
            }

            for (j = j0; j < j1; ++j, ++jp) {
              atomicAdd(vk + j + nao * k, p_buf_jk[jp]);
              atomicAdd(vk + j + nao * k + nao2, p_buf_jk[jp + nfjk]);
              atomicAdd(vk + j + nao * k + 2 * nao2, p_buf_jk[jp + 2 * nfjk]);
            }
          }

          for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
            for (i = i0; i < i1; ++i, ++ip) {
              atomicAdd(vk + i + nao * l, p_buf_il[ip]);
              atomicAdd(vk + i + nao * l + nao2, p_buf_il[ip + nfil]);
              atomicAdd(vk + i + nao * l + 2 * nao2, p_buf_il[ip + 2 * nfil]);
            }

            for (j = j0; j < j1; ++j, ++jp) {
              atomicAdd(vk + j + nao * l, p_buf_jl[jp]);
              atomicAdd(vk + j + nao * l + nao2, p_buf_jl[jp + nfjl]);
              atomicAdd(vk + j + nao * l + 2 * nao2, p_buf_jl[jp + 2 * nfjl]);
            }
          }
          vk += 3 * nao2;
          p_buf_il += 3 * nfil;
          p_buf_jl += 3 * nfjl;
          p_buf_ik += 3 * nfik;
          p_buf_jk += 3 * nfjk;
        }
      } else {
        extern __shared__ double buf_ik[];
        double * __restrict__ buf_il = buf_ik + 3 * nfik * n_dm * THREADS;
        double * __restrict__ buf_jk = gout;
        double * __restrict__ buf_jl = gout + 3 * nfjk * n_dm;

        memset(gout, 0, (3 * nfjk + 3 * nfjl) * n_dm * sizeof(double));

        for (ip = 0; ip < 3 * (nfik + nfil) * n_dm; ++ip) {
          buf_ik[ip * THREADS + task_id] = 0;
        }

        for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
          double ai = exponent_i[(ij - prim_ij) / nprim_j];
          double aj = exponent_j[(ij - prim_ij) % nprim_j];
          for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
            GINTg0_2e_2d4d<NROOTS>(g, uw, norm,
                                   as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
            dm = jk.dm;
            for (i_dm = 0; i_dm < n_dm; ++i_dm) {
              p_buf_il = buf_il + 3 * i_dm * nfil * THREADS;
              p_buf_jl = buf_jl + 3 * i_dm * nfjl;
              for (f = 0, l = l0; l < l1; ++l) {
                p_buf_ik = buf_ik + 3 * i_dm * nfik * THREADS;
                p_buf_jk = buf_jk + 3 * i_dm * nfjk;
                for (k = k0; k < k1; ++k) {
                  for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                    d_jl = dm[j + nao * l];
                    d_jk = dm[j + nao * k];

                    v_jl_x = 0;
                    v_jl_y = 0;
                    v_jl_z = 0;
                    v_jk_x = 0;
                    v_jk_y = 0;
                    v_jk_z = 0;

                    for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                      d_il = dm[i + nao * l];
                      d_ik = dm[i + nao * k];

                      GINTgout2e_nabla1i_per_function<NROOTS>(g, ai, aj, f,
                                                              &s_ix, &s_iy,
                                                              &s_iz,
                                                              &s_jx, &s_jy,
                                                              &s_jz);

                      p_buf_ik[ip * THREADS + task_id] += s_ix * d_jl;
                      p_buf_ik[(ip + nfik) * THREADS + task_id] += s_iy * d_jl;
                      p_buf_ik[(ip + 2 * nfik) * THREADS + task_id] += s_iz * d_jl;

                      p_buf_il[ip * THREADS + task_id] += s_ix * d_jk;
                      p_buf_il[(ip + nfil) * THREADS + task_id] += s_iy * d_jk;
                      p_buf_il[(ip + 2 * nfil) * THREADS + task_id] +=
                          s_iz * d_jk;

                      v_jl_x += s_jx * d_ik;
                      v_jl_y += s_jy * d_ik;
                      v_jl_z += s_jz * d_ik;

                      v_jk_x += s_jx * d_il;
                      v_jk_y += s_jy * d_il;
                      v_jk_z += s_jz * d_il;
                    }

                    p_buf_jl[jp] += v_jl_x;
                    p_buf_jl[jp + nfjl] += v_jl_y;
                    p_buf_jl[jp + 2 * nfjl] += v_jl_z;

                    p_buf_jk[jp] += v_jk_x;
                    p_buf_jk[jp + nfjk] += v_jk_y;
                    p_buf_jk[jp + 2 * nfjk] += v_jk_z;
                  }

                  p_buf_jk += nfj;
                  p_buf_ik += nfi * THREADS;
                }

                p_buf_il += nfi * THREADS;
                p_buf_jl += nfj;
              }
              dm += nao2;
            }
            uw += NROOTS * 2;
          }
        }

        p_buf_il = buf_il;
        p_buf_jl = buf_jl;
        p_buf_ik = buf_ik;
        p_buf_jk = buf_jk;
        for (i_dm = 0; i_dm < n_dm; ++i_dm) {

          for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
            for (i = i0; i < i1; ++i, ++ip) {
              atomicAdd(vk + i + nao * k, p_buf_ik[ip * THREADS + task_id]);
              atomicAdd(vk + i + nao * k + nao2,
                        p_buf_ik[(ip + nfik) * THREADS + task_id]);
              atomicAdd(vk + i + nao * k + 2 * nao2,
                        p_buf_ik[(ip + 2 * nfik) * THREADS + task_id]);
            }

            for (j = j0; j < j1; ++j, ++jp) {
              atomicAdd(vk + j + nao * k, p_buf_jk[jp]);
              atomicAdd(vk + j + nao * k + nao2, p_buf_jk[jp + nfjk]);
              atomicAdd(vk + j + nao * k + 2 * nao2, p_buf_jk[jp + 2 * nfjk]);
            }
          }

          for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
            for (i = i0; i < i1; ++i, ++ip) {
              atomicAdd(vk + i + nao * l, p_buf_il[ip * THREADS + task_id]);
              atomicAdd(vk + i + nao * l + nao2,
                        p_buf_il[(ip + nfil) * THREADS + task_id]);
              atomicAdd(vk + i + nao * l + 2 * nao2,
                        p_buf_il[(ip + 2 * nfil) * THREADS + task_id]);
            }

            for (j = j0; j < j1; ++j, ++jp) {
              atomicAdd(vk + j + nao * l, p_buf_jl[jp]);
              atomicAdd(vk + j + nao * l + nao2, p_buf_jl[jp + nfjl]);
              atomicAdd(vk + j + nao * l + 2 * nao2, p_buf_jl[jp + 2 * nfjl]);
            }
          }

          vk += 3 * nao2;
          p_buf_il += 3 * nfil * THREADS;
          p_buf_jl += 3 * nfjl;
          p_buf_ik += 3 * nfik * THREADS;
          p_buf_jk += 3 * nfjk;
        }
      }

    }
  }

}

__global__
static void
GINTint2e_jk_kernel_nabla1i_0000(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;
  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int * ao_loc = c_bpcache.ao_loc;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh];
  int l0 = ao_loc[lsh];

//  if(ish == jsh) {
//    norm *= 0.5;
//  }

  int nbas = c_bpcache.nbas;
  double * __restrict__ bas_x = c_bpcache.bas_coords;
  double * __restrict__ bas_y = bas_x + nbas;
  double * __restrict__ bas_z = bas_y + nbas;

  double xi = bas_x[ish];
  double yi = bas_y[ish];
  double zi = bas_z[ish];

  double xj = bas_x[jsh];
  double yj = bas_y[jsh];
  double zj = bas_z[jsh];

  double * __restrict__ a12 = c_bpcache.a12;
  double * __restrict__ e12 = c_bpcache.e12;
  double * __restrict__ x12 = c_bpcache.x12;
  double * __restrict__ y12 = c_bpcache.y12;
  double * __restrict__ z12 = c_bpcache.z12;
  int ij, kl, i_dm;
  double gout0 = 0, gout0_prime = 0;
  double gout1 = 0, gout1_prime = 0;
  double gout2 = 0, gout2_prime = 0;
  int nprim_j =
      c_bpcache.primitive_functions_offsets[jsh + 1]
      - c_bpcache.primitive_functions_offsets[jsh];

  double * exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];
  double * exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
    double ai = exponent_i[(ij - prim_ij) / nprim_j];
    double aj = exponent_j[(ij - prim_ij) % nprim_j];
    double aij = a12[ij];
    double eij = e12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
      double akl = a12[kl];
      double ekl = e12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      double fac = norm * eij * ekl / (sqrt(aijkl) * a1);

      double root0, weight0;

      if (x < 3.e-7) {
        root0 = 0.5;
        weight0 = 1.;
      } else {
        double tt = sqrt(x);
        double fmt0 = SQRTPIE4 / tt * erf(tt);
        weight0 = fmt0;
        double e = exp(-x);
        double b = .5 / x;
        double fmt1 = b * (fmt0 - e);
        root0 = fmt1 / (fmt0 - fmt1);
      }

      double u2 = a0 * root0;
      double tmp2 = akl * u2 / (u2 * aijkl + a1);
      double c00x = xij - xi - tmp2 * xijxkl;
      double c00y = yij - yi - tmp2 * yijykl;
      double c00z = zij - zi - tmp2 * zijzkl;

      double c00x_prime = xij - xj - tmp2 * xijxkl;
      double c00y_prime = yij - yj - tmp2 * yijykl;
      double c00z_prime = zij - zj - tmp2 * zijzkl;

      double g_0 = 1;
      double g_1 = c00x;
      double g_2 = 1;
      double g_3 = c00y;
      double g_4 = fac * weight0;
      double g_5 = g_4 * c00z;
      double g_6 = 2.0 * ai;

      double g_1_prime = c00x_prime;
      double g_3_prime = c00y_prime;
      double g_5_prime = g_4 * c00z_prime;
      double g_6_prime = 2.0 * aj;

      gout0 += g_1 * g_2 * g_4 * g_6;
      gout1 += g_0 * g_3 * g_4 * g_6;
      gout2 += g_0 * g_2 * g_5 * g_6;

      gout0_prime += g_1_prime * g_2 * g_4 * g_6_prime;
      gout1_prime += g_0 * g_3_prime * g_4 * g_6_prime;
      gout2_prime += g_0 * g_2 * g_5_prime * g_6_prime;
    }
  }

  int n_dm = jk.n_dm;
  int nao = jk.nao;
  size_t nao2 = nao * nao;
  double * __restrict__ dm = jk.dm;
  double * __restrict__ vj = jk.vj;
  double * __restrict__ vk = jk.vk;
  double d_0;

  for (i_dm = 0; i_dm < n_dm; ++i_dm) {
    if (vj != NULL) {
      d_0 = dm[k0 + nao * l0];
      atomicAdd(vj + i0 + nao * j0, gout0 * d_0);
      atomicAdd(vj + i0 + nao * j0 + nao2, gout1 * d_0);
      atomicAdd(vj + i0 + nao * j0 + 2 * nao2, gout2 * d_0);
      atomicAdd(vj + nao * i0 + j0, gout0_prime * d_0);
      atomicAdd(vj + nao * i0 + j0 + nao2, gout1_prime * d_0);
      atomicAdd(vj + nao * i0 + j0 + 2 * nao2, gout2_prime * d_0);
      vj += 3 * nao2;
    }
    if (vk != NULL) {
      // ijkl, jk -> il
      d_0 = dm[j0 + nao * k0];
      atomicAdd(vk + i0 + nao * l0, gout0 * d_0);
      atomicAdd(vk + i0 + nao * l0 + nao2, gout1 * d_0);
      atomicAdd(vk + i0 + nao * l0 + 2 * nao2, gout2 * d_0);
      // ijkl, jl -> ik
      d_0 = dm[j0 + nao * l0];
      atomicAdd(vk + i0 + nao * k0, gout0 * d_0);
      atomicAdd(vk + i0 + nao * k0 + nao2, gout1 * d_0);
      atomicAdd(vk + i0 + nao * k0 + 2 * nao2, gout2 * d_0);
      // ijkl, ik -> jl
      d_0 = dm[i0 + nao * k0];
      atomicAdd(vk + j0 + nao * l0, gout0_prime * d_0);
      atomicAdd(vk + j0 + nao * l0 + nao2, gout1_prime * d_0);
      atomicAdd(vk + j0 + nao * l0 + 2 * nao2, gout2_prime * d_0);
      // ijkl, il -> jk
      d_0 = dm[i0 + nao * l0];
      atomicAdd(vk + j0 + nao * k0, gout0_prime * d_0);
      atomicAdd(vk + j0 + nao * k0 + nao2, gout1_prime * d_0);
      atomicAdd(vk + j0 + nao * k0 + 2 * nao2, gout2_prime * d_0);
      vk += 3 * nao2;
    }
    dm += nao2;
  }
}

#if POLYFIT_ORDER >= 4

template<>
__global__
void GINTint2e_jk_kernel_nabla1i<4, NABLAGOUTSIZE4>(JKMatrix jk,
                                                    BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  double uw[8];
  double gout[NABLAGOUTSIZE4];
  double * g = gout + 6 * c_envs.nf;
  memset(gout, 0, 6 * c_envs.nf * sizeof(double));

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ a12 = c_bpcache.a12;
  double * __restrict__ x12 = c_bpcache.x12;
  double * __restrict__ y12 = c_bpcache.y12;
  double * __restrict__ z12 = c_bpcache.z12;
  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }
  for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
    double ai = exponent_i[(ij - prim_ij) / nprim_j];
    double aj = exponent_j[(ij - prim_ij) % nprim_j];
    double aij = a12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
      double akl = a12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      GINTrys_root4(x, uw);
      GINTg0_2e_2d4d<4>(g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
      GINTgout2e_nabla1i<4>(gout, g, ai, aj);
    }
  }

  GINTkernel_getjk_nabla1i(jk, gout, ish, jsh, ksh, lsh);
}

#endif

#if POLYFIT_ORDER >= 5

template<>
__global__
void GINTint2e_jk_kernel_nabla1i<5, NABLAGOUTSIZE5>(JKMatrix jk,
                                                    BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;
  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  double uw[10];
  double gout[NABLAGOUTSIZE5];
  double * g = gout + 6 * c_envs.nf;
  memset(gout, 0, 6 * c_envs.nf * sizeof(double));
  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ a12 = c_bpcache.a12;
  double * __restrict__ x12 = c_bpcache.x12;
  double * __restrict__ y12 = c_bpcache.y12;
  double * __restrict__ z12 = c_bpcache.z12;
  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }
  for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
    double ai = exponent_i[(ij - prim_ij) / nprim_j];
    double aj = exponent_j[(ij - prim_ij) % nprim_j];
    double aij = a12[ij];
    double xij = x12[ij];
    double yij = y12[ij];
    double zij = z12[ij];
    for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
      double akl = a12[kl];
      double xkl = x12[kl];
      double ykl = y12[kl];
      double zkl = z12[kl];
      double xijxkl = xij - xkl;
      double yijykl = yij - ykl;
      double zijzkl = zij - zkl;
      double aijkl = aij + akl;
      double a1 = aij * akl;
      double a0 = a1 / aijkl;
      double x = a0 * (xijxkl * xijxkl + yijykl * yijykl + zijzkl * zijzkl);
      GINTrys_root5(x, uw);
      GINTg0_2e_2d4d<5>(g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
      GINTgout2e_nabla1i<5>(gout, g, ai, aj);
    }
  }

  GINTkernel_getjk_nabla1i(jk, gout, ish, jsh, ksh, lsh);
}

#endif

__global__
static void GINTint2e_jk_kernel_nabla1i_3233_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 7 *
                             2;
  double gout[9660];
  memset(gout, 0, 9660 * sizeof(double));

  double * __restrict__ g = gout + 1260;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*60];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<60;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij_shared;
    double * __restrict__ p_buf_ij_global;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<7>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij_shared = shared_memory;
        p_buf_ij_global = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<7>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                p_buf_ij_global[n+0*nfij]+=s_iy * d_kl;
                p_buf_ij_global[n+1*nfij]+=s_iz * d_kl;
                p_buf_ij_global[n+2*nfij]+=s_jx * d_kl;
                p_buf_ij_global[n+3*nfij]+=s_jy * d_kl;
                p_buf_ij_global[n+4*nfij]+=s_jz * d_kl;
              }
            }
          }
        }
        uw += 7 * 2;
      }
    }

    p_buf_ij_shared = shared_memory;
    p_buf_ij_global = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_global[n+0*nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+1*nfij]);
        atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+2*nfij]);
        atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+3*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+4*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij_shared;
      double * __restrict__ p_buf_ij_global;

      double * buf_ik = gout + 300;
      double * buf_il = gout + 600;
      double * buf_jk = gout + 900;
      double * buf_jl = gout + 1080;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<7>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij_shared = shared_memory;
          p_buf_ij_global = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<7>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                  p_buf_ij_global[n+0*nfij]+=s_iy * d_kl;
                  p_buf_ij_global[n+1*nfij]+=s_iz * d_kl;
                  p_buf_ij_global[n+2*nfij]+=s_jx * d_kl;
                  p_buf_ij_global[n+3*nfij]+=s_jy * d_kl;
                  p_buf_ij_global[n+4*nfij]+=s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 7 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij_shared = shared_memory;
      p_buf_ij_global = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_global[n+0*nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+1*nfij]);
          atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+2*nfij]);
          atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+3*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+4*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_ik = gout + 0;
      double * buf_il = gout + 300;
      double * buf_jk = gout + 600;
      double * buf_jl = gout + 780;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<7>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<7>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 7 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3332_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 7 *
                             2;
  double gout[9960];
  memset(gout, 0, 9960 * sizeof(double));

  double * __restrict__ g = gout + 1560;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }



  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<7>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<7>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij[n       ] += s_ix * d_kl;
                p_buf_ij[n+  nfij] += s_iy * d_kl;
                p_buf_ij[n+2*nfij] += s_iz * d_kl;
                p_buf_ij[n+3*nfij] += s_jx * d_kl;
                p_buf_ij[n+4*nfij] += s_jy * d_kl;
                p_buf_ij[n+5*nfij] += s_jz * d_kl;
              }
            }
          }
        }
        uw += 7 * 2;
      }
    }

    p_buf_ij = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
        atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
        atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
        atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij;

      double * buf_ik = gout + 600;
      double * buf_il = gout + 900;
      double * buf_jk = gout + 1080;
      double * buf_jl = gout + 1380;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<7>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<7>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij[n       ] += s_ix * d_kl;
                  p_buf_ij[n+  nfij] += s_iy * d_kl;
                  p_buf_ij[n+2*nfij] += s_iz * d_kl;
                  p_buf_ij[n+3*nfij] += s_jx * d_kl;
                  p_buf_ij[n+4*nfij] += s_jy * d_kl;
                  p_buf_ij[n+5*nfij] += s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 7 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
          atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
          atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
          atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_ik = gout + 0;
      double * buf_il = gout + 300;
      double * buf_jk = gout + 480;
      double * buf_jl = gout + 780;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<7>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<7>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 7 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3333_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 7 *
                             2;
  double gout[10200];
  memset(gout, 0, 10200 * sizeof(double));

  double * __restrict__ g = gout + 1800;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }



  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<7>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<7>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij[n       ] += s_ix * d_kl;
                p_buf_ij[n+  nfij] += s_iy * d_kl;
                p_buf_ij[n+2*nfij] += s_iz * d_kl;
                p_buf_ij[n+3*nfij] += s_jx * d_kl;
                p_buf_ij[n+4*nfij] += s_jy * d_kl;
                p_buf_ij[n+5*nfij] += s_jz * d_kl;
              }
            }
          }
        }
        uw += 7 * 2;
      }
    }

    p_buf_ij = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
        atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
        atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
        atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij;

      double * buf_ik = gout + 600;
      double * buf_il = gout + 900;
      double * buf_jk = gout + 1200;
      double * buf_jl = gout + 1500;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<7>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<7>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij[n       ] += s_ix * d_kl;
                  p_buf_ij[n+  nfij] += s_iy * d_kl;
                  p_buf_ij[n+2*nfij] += s_iz * d_kl;
                  p_buf_ij[n+3*nfij] += s_jx * d_kl;
                  p_buf_ij[n+4*nfij] += s_jy * d_kl;
                  p_buf_ij[n+5*nfij] += s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 7 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
          atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
          atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
          atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_ik = gout + 0;
      double * buf_il = gout + 300;
      double * buf_jk = gout + 600;
      double * buf_jl = gout + 900;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<7>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<7>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 7 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}

__global__
static void GINTint2e_jk_kernel_nabla1i_2133_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5166];
  memset(gout, 0, 5166 * sizeof(double));

  double * __restrict__ g = gout + 558;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*90];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<90;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij_shared;
    double * __restrict__ p_buf_ij_global;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij_shared = shared_memory;
        p_buf_ij_global = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]+=s_iz * d_kl;
                p_buf_ij_shared[(n+3*nfij)*THREADS+task_id]+=s_jx * d_kl;
                p_buf_ij_shared[(n+4*nfij)*THREADS+task_id]+=s_jy * d_kl;
                p_buf_ij_global[n+0*nfij]+=s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij_shared = shared_memory;
    p_buf_ij_global = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]);
        atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_shared[(n+3*nfij)*THREADS+task_id]);
        atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_shared[(n+4*nfij)*THREADS+task_id]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+0*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij_shared;
      double * __restrict__ p_buf_ij_global;

      double * buf_ik = gout + 18;
      double * buf_il = gout + 198;
      double * buf_jk = gout + 378;
      double * buf_jl = gout + 468;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij_shared = shared_memory;
          p_buf_ij_global = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                  p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                  p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]+=s_iz * d_kl;
                  p_buf_ij_shared[(n+3*nfij)*THREADS+task_id]+=s_jx * d_kl;
                  p_buf_ij_shared[(n+4*nfij)*THREADS+task_id]+=s_jy * d_kl;
                  p_buf_ij_global[n+0*nfij]+=s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij_shared = shared_memory;
      p_buf_ij_global = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]);
          atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_shared[(n+3*nfij)*THREADS+task_id]);
          atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_shared[(n+4*nfij)*THREADS+task_id]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+0*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_jk = shared_memory + 0 * THREADS;
      double * buf_ik = gout + 0;
      double * buf_il = gout + 180;
      double * buf_jl = gout + 360;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[ jp        *THREADS+task_id] += v_jk_x;
                p_buf_jk[(jp+  nfjk)*THREADS+task_id] += v_jk_y;
                p_buf_jk[(jp+2*nfjk)*THREADS+task_id] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj * THREADS;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[ jp        *THREADS+task_id]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[(jp+  nfjk)*THREADS+task_id]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[(jp+2*nfjk)*THREADS+task_id]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_2232_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5328];
  memset(gout, 0, 5328 * sizeof(double));

  double * __restrict__ g = gout + 720;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*72];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<72;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij_shared;
    double * __restrict__ p_buf_ij_global;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij_shared = shared_memory;
        p_buf_ij_global = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                p_buf_ij_global[n+0*nfij]+=s_iz * d_kl;
                p_buf_ij_global[n+1*nfij]+=s_jx * d_kl;
                p_buf_ij_global[n+2*nfij]+=s_jy * d_kl;
                p_buf_ij_global[n+3*nfij]+=s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij_shared = shared_memory;
    p_buf_ij_global = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+0*nfij]);
        atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+1*nfij]);
        atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+2*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+3*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij_shared;
      double * __restrict__ p_buf_ij_global;

      double * buf_ik = gout + 144;
      double * buf_il = gout + 324;
      double * buf_jk = gout + 432;
      double * buf_jl = gout + 612;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij_shared = shared_memory;
          p_buf_ij_global = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                  p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                  p_buf_ij_global[n+0*nfij]+=s_iz * d_kl;
                  p_buf_ij_global[n+1*nfij]+=s_jx * d_kl;
                  p_buf_ij_global[n+2*nfij]+=s_jy * d_kl;
                  p_buf_ij_global[n+3*nfij]+=s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij_shared = shared_memory;
      p_buf_ij_global = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+0*nfij]);
          atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+1*nfij]);
          atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+2*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+3*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_ik = gout + 0;
      double * buf_il = gout + 180;
      double * buf_jk = gout + 288;
      double * buf_jl = gout + 468;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_2233_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5472];
  memset(gout, 0, 5472 * sizeof(double));

  double * __restrict__ g = gout + 864;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*72];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<72;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij_shared;
    double * __restrict__ p_buf_ij_global;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij_shared = shared_memory;
        p_buf_ij_global = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                p_buf_ij_global[n+0*nfij]+=s_iz * d_kl;
                p_buf_ij_global[n+1*nfij]+=s_jx * d_kl;
                p_buf_ij_global[n+2*nfij]+=s_jy * d_kl;
                p_buf_ij_global[n+3*nfij]+=s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij_shared = shared_memory;
    p_buf_ij_global = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+0*nfij]);
        atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+1*nfij]);
        atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+2*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+3*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij_shared;
      double * __restrict__ p_buf_ij_global;

      double * buf_ik = gout + 144;
      double * buf_il = gout + 324;
      double * buf_jk = gout + 504;
      double * buf_jl = gout + 684;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij_shared = shared_memory;
          p_buf_ij_global = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                  p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                  p_buf_ij_global[n+0*nfij]+=s_iz * d_kl;
                  p_buf_ij_global[n+1*nfij]+=s_jx * d_kl;
                  p_buf_ij_global[n+2*nfij]+=s_jy * d_kl;
                  p_buf_ij_global[n+3*nfij]+=s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij_shared = shared_memory;
      p_buf_ij_global = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+0*nfij]);
          atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+1*nfij]);
          atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+2*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+3*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_ik = gout + 0;
      double * buf_il = gout + 180;
      double * buf_jk = gout + 360;
      double * buf_jl = gout + 540;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3033_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5238];
  memset(gout, 0, 5238 * sizeof(double));

  double * __restrict__ g = gout + 630;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*90];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<90;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij = shared_memory;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij[ n        *THREADS+task_id] += s_ix * d_kl;
                p_buf_ij[(n+  nfij)*THREADS+task_id] += s_iy * d_kl;
                p_buf_ij[(n+2*nfij)*THREADS+task_id] += s_iz * d_kl;
                p_buf_ij[(n+3*nfij)*THREADS+task_id] += s_jx * d_kl;
                p_buf_ij[(n+4*nfij)*THREADS+task_id] += s_jy * d_kl;
                p_buf_ij[(n+5*nfij)*THREADS+task_id] += s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij = shared_memory;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j       , p_buf_ij[ n        *THREADS+task_id]);
        atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[(n+  nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[(n+2*nfij)*THREADS+task_id]);
        atomicAdd(vj+j+nao*i       , p_buf_ij[(n+3*nfij)*THREADS+task_id]);
        atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[(n+4*nfij)*THREADS+task_id]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[(n+5*nfij)*THREADS+task_id]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij;

      double * buf_jk = shared_memory + 60 * THREADS;
      double * buf_ik = gout + 0;
      double * buf_il = gout + 300;
      double * buf_jl = gout + 600;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij = shared_memory;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij[ n        *THREADS+task_id] += s_ix * d_kl;
                  p_buf_ij[(n+  nfij)*THREADS+task_id] += s_iy * d_kl;
                  p_buf_ij[(n+2*nfij)*THREADS+task_id] += s_iz * d_kl;
                  p_buf_ij[(n+3*nfij)*THREADS+task_id] += s_jx * d_kl;
                  p_buf_ij[(n+4*nfij)*THREADS+task_id] += s_jy * d_kl;
                  p_buf_ij[(n+5*nfij)*THREADS+task_id] += s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[ jp        *THREADS+task_id] += v_jk_x;
                p_buf_jk[(jp+  nfjk)*THREADS+task_id] += v_jk_y;
                p_buf_jk[(jp+2*nfjk)*THREADS+task_id] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj * THREADS;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij = shared_memory;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j       , p_buf_ij[ n        *THREADS+task_id]);
          atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[(n+  nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[(n+2*nfij)*THREADS+task_id]);
          atomicAdd(vj+j+nao*i       , p_buf_ij[(n+3*nfij)*THREADS+task_id]);
          atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[(n+4*nfij)*THREADS+task_id]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[(n+5*nfij)*THREADS+task_id]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[ jp        *THREADS+task_id]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[(jp+  nfjk)*THREADS+task_id]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[(jp+2*nfjk)*THREADS+task_id]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_jk = shared_memory + 0 * THREADS;
      double * buf_jl = shared_memory + 30 * THREADS;
      double * buf_ik = gout + 0;
      double * buf_il = gout + 300;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[ jp        *THREADS+task_id] += v_jk_x;
                p_buf_jk[(jp+  nfjk)*THREADS+task_id] += v_jk_y;
                p_buf_jk[(jp+2*nfjk)*THREADS+task_id] += v_jk_z;

                p_buf_jl[ jp        *THREADS+task_id] += v_jl_x;
                p_buf_jl[(jp+  nfjl)*THREADS+task_id] += v_jl_y;
                p_buf_jl[(jp+2*nfjl)*THREADS+task_id] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj * THREADS;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj * THREADS;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[ jp        *THREADS+task_id]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[(jp+  nfjk)*THREADS+task_id]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[(jp+2*nfjk)*THREADS+task_id]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[ jp        *THREADS+task_id]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[(jp+  nfjl)*THREADS+task_id]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[(jp+2*nfjl)*THREADS+task_id]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3132_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5322];
  memset(gout, 0, 5322 * sizeof(double));

  double * __restrict__ g = gout + 714;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*90];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<90;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij_shared;
    double * __restrict__ p_buf_ij_global;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij_shared = shared_memory;
        p_buf_ij_global = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]+=s_iz * d_kl;
                p_buf_ij_global[n+0*nfij]+=s_jx * d_kl;
                p_buf_ij_global[n+1*nfij]+=s_jy * d_kl;
                p_buf_ij_global[n+2*nfij]+=s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij_shared = shared_memory;
    p_buf_ij_global = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]);
        atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+0*nfij]);
        atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+1*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+2*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij_shared;
      double * __restrict__ p_buf_ij_global;

      double * buf_ik = gout + 90;
      double * buf_il = gout + 390;
      double * buf_jk = gout + 570;
      double * buf_jl = gout + 660;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij_shared = shared_memory;
          p_buf_ij_global = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                  p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                  p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]+=s_iz * d_kl;
                  p_buf_ij_global[n+0*nfij]+=s_jx * d_kl;
                  p_buf_ij_global[n+1*nfij]+=s_jy * d_kl;
                  p_buf_ij_global[n+2*nfij]+=s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij_shared = shared_memory;
      p_buf_ij_global = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]);
          atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+0*nfij]);
          atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+1*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+2*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_jk = shared_memory + 0 * THREADS;
      double * buf_ik = gout + 0;
      double * buf_il = gout + 300;
      double * buf_jl = gout + 480;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[ jp        *THREADS+task_id] += v_jk_x;
                p_buf_jk[(jp+  nfjk)*THREADS+task_id] += v_jk_y;
                p_buf_jk[(jp+2*nfjk)*THREADS+task_id] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj * THREADS;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[ jp        *THREADS+task_id]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[(jp+  nfjk)*THREADS+task_id]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[(jp+2*nfjk)*THREADS+task_id]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3133_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5478];
  memset(gout, 0, 5478 * sizeof(double));

  double * __restrict__ g = gout + 870;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*90];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<90;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij_shared;
    double * __restrict__ p_buf_ij_global;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij_shared = shared_memory;
        p_buf_ij_global = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]+=s_iz * d_kl;
                p_buf_ij_global[n+0*nfij]+=s_jx * d_kl;
                p_buf_ij_global[n+1*nfij]+=s_jy * d_kl;
                p_buf_ij_global[n+2*nfij]+=s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij_shared = shared_memory;
    p_buf_ij_global = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]);
        atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+0*nfij]);
        atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+1*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+2*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij_shared;
      double * __restrict__ p_buf_ij_global;

      double * buf_ik = gout + 90;
      double * buf_il = gout + 390;
      double * buf_jk = gout + 690;
      double * buf_jl = gout + 780;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij_shared = shared_memory;
          p_buf_ij_global = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                  p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]+=s_iy * d_kl;
                  p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]+=s_iz * d_kl;
                  p_buf_ij_global[n+0*nfij]+=s_jx * d_kl;
                  p_buf_ij_global[n+1*nfij]+=s_jy * d_kl;
                  p_buf_ij_global[n+2*nfij]+=s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij_shared = shared_memory;
      p_buf_ij_global = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_shared[(n+1*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_shared[(n+2*nfij)*THREADS+task_id]);
          atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+0*nfij]);
          atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+1*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+2*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_jk = shared_memory + 0 * THREADS;
      double * buf_ik = gout + 0;
      double * buf_il = gout + 300;
      double * buf_jl = gout + 600;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[ jp        *THREADS+task_id] += v_jk_x;
                p_buf_jk[(jp+  nfjk)*THREADS+task_id] += v_jk_y;
                p_buf_jk[(jp+2*nfjk)*THREADS+task_id] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj * THREADS;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[ jp        *THREADS+task_id]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[(jp+  nfjk)*THREADS+task_id]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[(jp+2*nfjk)*THREADS+task_id]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3222_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5484];
  memset(gout, 0, 5484 * sizeof(double));

  double * __restrict__ g = gout + 876;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*60];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<60;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij_shared;
    double * __restrict__ p_buf_ij_global;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij_shared = shared_memory;
        p_buf_ij_global = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                p_buf_ij_global[n+0*nfij]+=s_iy * d_kl;
                p_buf_ij_global[n+1*nfij]+=s_iz * d_kl;
                p_buf_ij_global[n+2*nfij]+=s_jx * d_kl;
                p_buf_ij_global[n+3*nfij]+=s_jy * d_kl;
                p_buf_ij_global[n+4*nfij]+=s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij_shared = shared_memory;
    p_buf_ij_global = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_global[n+0*nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+1*nfij]);
        atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+2*nfij]);
        atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+3*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+4*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij_shared;
      double * __restrict__ p_buf_ij_global;

      double * buf_ik = gout + 300;
      double * buf_il = gout + 480;
      double * buf_jk = gout + 660;
      double * buf_jl = gout + 768;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij_shared = shared_memory;
          p_buf_ij_global = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                  p_buf_ij_global[n+0*nfij]+=s_iy * d_kl;
                  p_buf_ij_global[n+1*nfij]+=s_iz * d_kl;
                  p_buf_ij_global[n+2*nfij]+=s_jx * d_kl;
                  p_buf_ij_global[n+3*nfij]+=s_jy * d_kl;
                  p_buf_ij_global[n+4*nfij]+=s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij_shared = shared_memory;
      p_buf_ij_global = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_global[n+0*nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+1*nfij]);
          atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+2*nfij]);
          atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+3*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+4*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_ik = gout + 0;
      double * buf_il = gout + 180;
      double * buf_jk = gout + 360;
      double * buf_jl = gout + 468;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3231_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5532];
  memset(gout, 0, 5532 * sizeof(double));

  double * __restrict__ g = gout + 924;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*60];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<60;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij_shared;
    double * __restrict__ p_buf_ij_global;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij_shared = shared_memory;
        p_buf_ij_global = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                p_buf_ij_global[n+0*nfij]+=s_iy * d_kl;
                p_buf_ij_global[n+1*nfij]+=s_iz * d_kl;
                p_buf_ij_global[n+2*nfij]+=s_jx * d_kl;
                p_buf_ij_global[n+3*nfij]+=s_jy * d_kl;
                p_buf_ij_global[n+4*nfij]+=s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij_shared = shared_memory;
    p_buf_ij_global = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_global[n+0*nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+1*nfij]);
        atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+2*nfij]);
        atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+3*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+4*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij_shared;
      double * __restrict__ p_buf_ij_global;

      double * buf_ik = gout + 300;
      double * buf_il = gout + 600;
      double * buf_jk = gout + 690;
      double * buf_jl = gout + 870;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij_shared = shared_memory;
          p_buf_ij_global = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                  p_buf_ij_global[n+0*nfij]+=s_iy * d_kl;
                  p_buf_ij_global[n+1*nfij]+=s_iz * d_kl;
                  p_buf_ij_global[n+2*nfij]+=s_jx * d_kl;
                  p_buf_ij_global[n+3*nfij]+=s_jy * d_kl;
                  p_buf_ij_global[n+4*nfij]+=s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij_shared = shared_memory;
      p_buf_ij_global = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_global[n+0*nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+1*nfij]);
          atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+2*nfij]);
          atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+3*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+4*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_il = shared_memory + 0 * THREADS;
      double * buf_ik = gout + 0;
      double * buf_jk = gout + 300;
      double * buf_jl = gout + 480;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ ip        *THREADS+task_id] += s_ix * d_jk;
                  p_buf_il[(ip+  nfil)*THREADS+task_id] += s_iy * d_jk;
                  p_buf_il[(ip+2*nfil)*THREADS+task_id] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi * THREADS;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ ip        *THREADS+task_id]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[(ip+  nfil)*THREADS+task_id]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[(ip+2*nfil)*THREADS+task_id]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3232_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5676];
  memset(gout, 0, 5676 * sizeof(double));

  double * __restrict__ g = gout + 1068;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*60];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<60;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij_shared;
    double * __restrict__ p_buf_ij_global;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij_shared = shared_memory;
        p_buf_ij_global = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                p_buf_ij_global[n+0*nfij]+=s_iy * d_kl;
                p_buf_ij_global[n+1*nfij]+=s_iz * d_kl;
                p_buf_ij_global[n+2*nfij]+=s_jx * d_kl;
                p_buf_ij_global[n+3*nfij]+=s_jy * d_kl;
                p_buf_ij_global[n+4*nfij]+=s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij_shared = shared_memory;
    p_buf_ij_global = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
        atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_global[n+0*nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+1*nfij]);
        atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+2*nfij]);
        atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+3*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+4*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij_shared;
      double * __restrict__ p_buf_ij_global;

      double * buf_ik = gout + 300;
      double * buf_il = gout + 600;
      double * buf_jk = gout + 780;
      double * buf_jl = gout + 960;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij_shared = shared_memory;
          p_buf_ij_global = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]+=s_ix * d_kl;
                  p_buf_ij_global[n+0*nfij]+=s_iy * d_kl;
                  p_buf_ij_global[n+1*nfij]+=s_iz * d_kl;
                  p_buf_ij_global[n+2*nfij]+=s_jx * d_kl;
                  p_buf_ij_global[n+3*nfij]+=s_jy * d_kl;
                  p_buf_ij_global[n+4*nfij]+=s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij_shared = shared_memory;
      p_buf_ij_global = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j+0*nao2, p_buf_ij_shared[(n+0*nfij)*THREADS+task_id]);
          atomicAdd(vj+i+nao*j+1*nao2, p_buf_ij_global[n+0*nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij_global[n+1*nfij]);
          atomicAdd(vj+j+nao*i+0*nao2, p_buf_ij_global[n+2*nfij]);
          atomicAdd(vj+j+nao*i+1*nao2, p_buf_ij_global[n+3*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij_global[n+4*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_ik = gout + 0;
      double * buf_il = gout + 300;
      double * buf_jk = gout + 480;
      double * buf_jl = gout + 660;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3321_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5658];
  memset(gout, 0, 5658 * sizeof(double));

  double * __restrict__ g = gout + 1050;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*90];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<90;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij[n       ] += s_ix * d_kl;
                p_buf_ij[n+  nfij] += s_iy * d_kl;
                p_buf_ij[n+2*nfij] += s_iz * d_kl;
                p_buf_ij[n+3*nfij] += s_jx * d_kl;
                p_buf_ij[n+4*nfij] += s_jy * d_kl;
                p_buf_ij[n+5*nfij] += s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
        atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
        atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
        atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij;

      double * buf_il = shared_memory + 0 * THREADS;
      double * buf_ik = gout + 600;
      double * buf_jk = gout + 780;
      double * buf_jl = gout + 960;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij[n       ] += s_ix * d_kl;
                  p_buf_ij[n+  nfij] += s_iy * d_kl;
                  p_buf_ij[n+2*nfij] += s_iz * d_kl;
                  p_buf_ij[n+3*nfij] += s_jx * d_kl;
                  p_buf_ij[n+4*nfij] += s_jy * d_kl;
                  p_buf_ij[n+5*nfij] += s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ ip        *THREADS+task_id] += s_ix * d_jk;
                  p_buf_il[(ip+  nfil)*THREADS+task_id] += s_iy * d_jk;
                  p_buf_il[(ip+2*nfil)*THREADS+task_id] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi * THREADS;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
          atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
          atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
          atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ ip        *THREADS+task_id]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[(ip+  nfil)*THREADS+task_id]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[(ip+2*nfil)*THREADS+task_id]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_il = shared_memory + 0 * THREADS;
      double * buf_ik = gout + 0;
      double * buf_jk = gout + 180;
      double * buf_jl = gout + 360;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ ip        *THREADS+task_id] += s_ix * d_jk;
                  p_buf_il[(ip+  nfil)*THREADS+task_id] += s_iy * d_jk;
                  p_buf_il[(ip+2*nfil)*THREADS+task_id] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi * THREADS;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ ip        *THREADS+task_id]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[(ip+  nfil)*THREADS+task_id]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[(ip+2*nfil)*THREADS+task_id]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3322_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5928];
  memset(gout, 0, 5928 * sizeof(double));

  double * __restrict__ g = gout + 1320;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }



  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij[n       ] += s_ix * d_kl;
                p_buf_ij[n+  nfij] += s_iy * d_kl;
                p_buf_ij[n+2*nfij] += s_iz * d_kl;
                p_buf_ij[n+3*nfij] += s_jx * d_kl;
                p_buf_ij[n+4*nfij] += s_jy * d_kl;
                p_buf_ij[n+5*nfij] += s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
        atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
        atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
        atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij;

      double * buf_ik = gout + 600;
      double * buf_il = gout + 780;
      double * buf_jk = gout + 960;
      double * buf_jl = gout + 1140;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij[n       ] += s_ix * d_kl;
                  p_buf_ij[n+  nfij] += s_iy * d_kl;
                  p_buf_ij[n+2*nfij] += s_iz * d_kl;
                  p_buf_ij[n+3*nfij] += s_jx * d_kl;
                  p_buf_ij[n+4*nfij] += s_jy * d_kl;
                  p_buf_ij[n+5*nfij] += s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
          atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
          atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
          atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_ik = gout + 0;
      double * buf_il = gout + 180;
      double * buf_jk = gout + 360;
      double * buf_jl = gout + 540;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ip       ] += s_ix * d_jk;
                  p_buf_il[ip+  nfil] += s_iy * d_jk;
                  p_buf_il[ip+2*nfil] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ip       ]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[ip+  nfil]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[ip+2*nfil]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3330_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5808];
  memset(gout, 0, 5808 * sizeof(double));

  double * __restrict__ g = gout + 1200;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*60];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<60;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij[n       ] += s_ix * d_kl;
                p_buf_ij[n+  nfij] += s_iy * d_kl;
                p_buf_ij[n+2*nfij] += s_iz * d_kl;
                p_buf_ij[n+3*nfij] += s_jx * d_kl;
                p_buf_ij[n+4*nfij] += s_jy * d_kl;
                p_buf_ij[n+5*nfij] += s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
        atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
        atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
        atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij;

      double * buf_il = shared_memory + 0 * THREADS;
      double * buf_jl = shared_memory + 30 * THREADS;
      double * buf_ik = gout + 600;
      double * buf_jk = gout + 900;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij[n       ] += s_ix * d_kl;
                  p_buf_ij[n+  nfij] += s_iy * d_kl;
                  p_buf_ij[n+2*nfij] += s_iz * d_kl;
                  p_buf_ij[n+3*nfij] += s_jx * d_kl;
                  p_buf_ij[n+4*nfij] += s_jy * d_kl;
                  p_buf_ij[n+5*nfij] += s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ ip        *THREADS+task_id] += s_ix * d_jk;
                  p_buf_il[(ip+  nfil)*THREADS+task_id] += s_iy * d_jk;
                  p_buf_il[(ip+2*nfil)*THREADS+task_id] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[ jp        *THREADS+task_id] += v_jl_x;
                p_buf_jl[(jp+  nfjl)*THREADS+task_id] += v_jl_y;
                p_buf_jl[(jp+2*nfjl)*THREADS+task_id] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi * THREADS;
            p_buf_jl += nfj * THREADS;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
          atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
          atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
          atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ ip        *THREADS+task_id]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[(ip+  nfil)*THREADS+task_id]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[(ip+2*nfil)*THREADS+task_id]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[ jp        *THREADS+task_id]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[(jp+  nfjl)*THREADS+task_id]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[(jp+2*nfjl)*THREADS+task_id]);
        }
      }
    } else { // only vk required

      double * buf_il = shared_memory + 0 * THREADS;
      double * buf_jl = shared_memory + 30 * THREADS;
      double * buf_ik = gout + 0;
      double * buf_jk = gout + 300;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ ip        *THREADS+task_id] += s_ix * d_jk;
                  p_buf_il[(ip+  nfil)*THREADS+task_id] += s_iy * d_jk;
                  p_buf_il[(ip+2*nfil)*THREADS+task_id] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[ jp        *THREADS+task_id] += v_jl_x;
                p_buf_jl[(jp+  nfjl)*THREADS+task_id] += v_jl_y;
                p_buf_jl[(jp+2*nfjl)*THREADS+task_id] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi * THREADS;
            p_buf_jl += nfj * THREADS;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ ip        *THREADS+task_id]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[(ip+  nfil)*THREADS+task_id]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[(ip+2*nfil)*THREADS+task_id]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[ jp        *THREADS+task_id]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[(jp+  nfjl)*THREADS+task_id]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[(jp+2*nfjl)*THREADS+task_id]);
        }
      }
    }
  }

}
__global__
static void GINTint2e_jk_kernel_nabla1i_3331_restricted(JKMatrix jk, BasisProdOffsets offsets) {
  int ntasks_ij = offsets.ntasks_ij;
  int ntasks_kl = offsets.ntasks_kl;
  int task_ij = blockIdx.x * blockDim.x + threadIdx.x;
  int task_kl = blockIdx.y * blockDim.y + threadIdx.y;

  if (task_ij >= ntasks_ij || task_kl >= ntasks_kl) {
    return;
  }

  int * ao_loc = c_bpcache.ao_loc;
  int bas_ij = offsets.bas_ij + task_ij;
  int bas_kl = offsets.bas_kl + task_kl;

  int nao = jk.nao;
  int nao2 = nao * nao;

  int i, j, k, l, n, f;
  int ip, jp;
  double d_kl, d_jk, d_jl, d_ik, d_il;
  double v_jk_x, v_jk_y, v_jk_z, v_jl_x, v_jl_y, v_jl_z;

  double norm = c_envs.fac;

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh];
  int k1 = ao_loc[ksh + 1];
  int l0 = ao_loc[lsh];
  int l1 = ao_loc[lsh + 1];
  int nfi = i1 - i0;
  int nfj = j1 - j0;
  int nfk = k1 - k0;
  int nfl = l1 - l0;
  int nfij = nfi * nfj;
  int nfik = nfi * nfk;
  int nfil = nfi * nfl;
  int nfjk = nfj * nfk;
  int nfjl = nfj * nfl;

  double * vj = jk.vj;
  double * vk = jk.vk;
  double * dm = jk.dm;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

  double * __restrict__ uw = c_envs.uw +
                             (task_ij + ntasks_ij * task_kl) * nprim_ij * nprim_kl * 6 *
                             2;
  double gout[5898];
  memset(gout, 0, 5898 * sizeof(double));

  double * __restrict__ g = gout + 1290;

  int nprim_j = c_bpcache.primitive_functions_offsets[jsh + 1]
                - c_bpcache.primitive_functions_offsets[jsh];

  double * __restrict__ exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];

  double * __restrict__ exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int ij, kl;
  int as_ish, as_jsh, as_ksh, as_lsh;
  if (c_envs.ibase) {
    as_ish = ish;
    as_jsh = jsh;
  } else {
    as_ish = jsh;
    as_jsh = ish;
  }
  if (c_envs.kbase) {
    as_ksh = ksh;
    as_lsh = lsh;
  } else {
    as_ksh = lsh;
    as_lsh = ksh;
  }

  __shared__ double shared_memory[THREADS*90];
  int task_id = threadIdx.y * THREADSX + threadIdx.x;
  for(ip=0;ip<90;++ip) {
    shared_memory[ip*THREADS+task_id]=0;
  }

  if (vk == NULL) {

    if (vj == NULL) {
      return;
    }

    double * __restrict__ p_buf_ij;
    for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
        GINTg0_2e_2d4d<6>(g, uw, norm,
                          as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
        p_buf_ij = gout;

        for (f = 0, l = l0; l < l1; ++l) {
          for (k = k0; k < k1; ++k) {
            d_kl = dm[k + nao * l];
            for (n = 0, j = j0; j < j1; ++j) {
              for (i = i0; i < i1; ++i, ++n, ++f) {
                GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                   &s_ix, &s_iy, &s_iz,
                                                   &s_jx, &s_jy,
                                                   &s_jz);
                p_buf_ij[n       ] += s_ix * d_kl;
                p_buf_ij[n+  nfij] += s_iy * d_kl;
                p_buf_ij[n+2*nfij] += s_iz * d_kl;
                p_buf_ij[n+3*nfij] += s_jx * d_kl;
                p_buf_ij[n+4*nfij] += s_jy * d_kl;
                p_buf_ij[n+5*nfij] += s_jz * d_kl;
              }
            }
          }
        }
        uw += 6 * 2;
      }
    }

    p_buf_ij = gout;
    for (n = 0, j = j0; j < j1; ++j) {
      for (i = i0; i < i1; ++i, ++n) {
        atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
        atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
        atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
        atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
        atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
        atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
      }
    }

  } else { // vk != NULL

    double * __restrict__ p_buf_ik;
    double * __restrict__ p_buf_jk;
    double * __restrict__ p_buf_il;
    double * __restrict__ p_buf_jl;

    if (vj != NULL) {
      double * __restrict__ p_buf_ij;

      double * buf_il = shared_memory + 0 * THREADS;
      double * buf_ik = gout + 600;
      double * buf_jk = gout + 900;
      double * buf_jl = gout + 1200;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);
          p_buf_ij = gout;
          p_buf_il = buf_il;
          p_buf_jl = buf_jl;
          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              d_kl = dm[k + nao * l];
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);
                  p_buf_ij[n       ] += s_ix * d_kl;
                  p_buf_ij[n+  nfij] += s_iy * d_kl;
                  p_buf_ij[n+2*nfij] += s_iz * d_kl;
                  p_buf_ij[n+3*nfij] += s_jx * d_kl;
                  p_buf_ij[n+4*nfij] += s_jy * d_kl;
                  p_buf_ij[n+5*nfij] += s_jz * d_kl;

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ ip        *THREADS+task_id] += s_ix * d_jk;
                  p_buf_il[(ip+  nfil)*THREADS+task_id] += s_iy * d_jk;
                  p_buf_il[(ip+2*nfil)*THREADS+task_id] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }

                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }

              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }

            p_buf_il += nfi * THREADS;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;
      p_buf_ij = gout;

      for (n = 0, j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          atomicAdd(vj+i+nao*j       , p_buf_ij[n       ]);
          atomicAdd(vj+i+nao*j+  nao2, p_buf_ij[n+  nfij]);
          atomicAdd(vj+i+nao*j+2*nao2, p_buf_ij[n+2*nfij]);
          atomicAdd(vj+j+nao*i       , p_buf_ij[n+3*nfij]);
          atomicAdd(vj+j+nao*i+  nao2, p_buf_ij[n+4*nfij]);
          atomicAdd(vj+j+nao*i+2*nao2, p_buf_ij[n+5*nfij]);
        }
      }

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ ip        *THREADS+task_id]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[(ip+  nfil)*THREADS+task_id]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[(ip+2*nfil)*THREADS+task_id]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    } else { // only vk required

      double * buf_il = shared_memory + 0 * THREADS;
      double * buf_ik = gout + 0;
      double * buf_jk = gout + 300;
      double * buf_jl = gout + 600;

      for (ij = prim_ij; ij < prim_ij + nprim_ij; ++ij) {
        double ai = exponent_i[(ij - prim_ij) / nprim_j];
        double aj = exponent_j[(ij - prim_ij) % nprim_j];
        for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
          GINTg0_2e_2d4d<6>(g, uw, norm,
                            as_ish, as_jsh, as_ksh, as_lsh, ij, kl);

          p_buf_il = buf_il;
          p_buf_jl = buf_jl;

          for (f = 0, l = l0; l < l1; ++l) {
            p_buf_ik = buf_ik;
            p_buf_jk = buf_jk;
            for (k = k0; k < k1; ++k) {
              for (jp = 0, n = 0, j = j0; j < j1; ++j, ++jp) {
                d_jl = dm[j + nao * l];
                d_jk = dm[j + nao * k];

                v_jl_x = 0;
                v_jl_y = 0;
                v_jl_z = 0;
                v_jk_x = 0;
                v_jk_y = 0;
                v_jk_z = 0;

                for (ip = 0, i = i0; i < i1; ++i, ++n, ++ip, ++f) {
                  d_il = dm[i + nao * l];
                  d_ik = dm[i + nao * k];

                  GINTgout2e_nabla1i_per_function<6>(g, ai, aj, f,
                                                     &s_ix, &s_iy,
                                                     &s_iz,
                                                     &s_jx, &s_jy,
                                                     &s_jz);

                  p_buf_ik[ip       ] += s_ix * d_jl;
                  p_buf_ik[ip+  nfik] += s_iy * d_jl;
                  p_buf_ik[ip+2*nfik] += s_iz * d_jl;

                  p_buf_il[ ip        *THREADS+task_id] += s_ix * d_jk;
                  p_buf_il[(ip+  nfil)*THREADS+task_id] += s_iy * d_jk;
                  p_buf_il[(ip+2*nfil)*THREADS+task_id] += s_iz * d_jk;

                  v_jl_x += s_jx * d_ik;
                  v_jl_y += s_jy * d_ik;
                  v_jl_z += s_jz * d_ik;

                  v_jk_x += s_jx * d_il;
                  v_jk_y += s_jy * d_il;
                  v_jk_z += s_jz * d_il;
                }
                p_buf_jk[jp       ] += v_jk_x;
                p_buf_jk[jp+  nfjk] += v_jk_y;
                p_buf_jk[jp+2*nfjk] += v_jk_z;

                p_buf_jl[jp       ] += v_jl_x;
                p_buf_jl[jp+  nfjl] += v_jl_y;
                p_buf_jl[jp+2*nfjl] += v_jl_z;
              }
              p_buf_ik += nfi;
              p_buf_jk += nfj;
            }
            p_buf_il += nfi * THREADS;
            p_buf_jl += nfj;
          }
          uw += 6 * 2;
        }
      }

      p_buf_il = buf_il;
      p_buf_jl = buf_jl;
      p_buf_ik = buf_ik;
      p_buf_jk = buf_jk;

      for (ip = 0, jp = 0, k = k0; k < k1; ++k) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*k       , p_buf_ik[ip       ]);
          atomicAdd(vk+i+nao*k+  nao2, p_buf_ik[ip+  nfik]);
          atomicAdd(vk+i+nao*k+2*nao2, p_buf_ik[ip+2*nfik]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*k       , p_buf_jk[jp       ]);
          atomicAdd(vk+j+nao*k+  nao2, p_buf_jk[jp+  nfjk]);
          atomicAdd(vk+j+nao*k+2*nao2, p_buf_jk[jp+2*nfjk]);
        }
      }

      for (ip = 0, jp = 0, l = l0; l < l1; ++l) {
        for (i = i0; i < i1; ++i, ++ip) {
          atomicAdd(vk+i+nao*l       , p_buf_il[ ip        *THREADS+task_id]);
          atomicAdd(vk+i+nao*l+  nao2, p_buf_il[(ip+  nfil)*THREADS+task_id]);
          atomicAdd(vk+i+nao*l+2*nao2, p_buf_il[(ip+2*nfil)*THREADS+task_id]);
        }

        for (j = j0; j < j1; ++j, ++jp) {
          atomicAdd(vk+j+nao*l       , p_buf_jl[jp       ]);
          atomicAdd(vk+j+nao*l+  nao2, p_buf_jl[jp+  nfjl]);
          atomicAdd(vk+j+nao*l+2*nao2, p_buf_jl[jp+2*nfjl]);
        }
      }
    }
  }

}