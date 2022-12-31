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

#include "nr_fill_ao_ints.cuh"

#include "g2e.h"
#include "gout2e.cuh"
#include "cint2e.cuh"
#include "rys_roots.cuh"
#include "fill_ints.cuh"

#define POLYFIT_ORDER   5
#define SQRTPIE4        .8862269254527580136
#define PIE4            .7853981633974483096

template<int NROOTS>
__device__
void GINTg0_2e_2d4d_nabla1i(double * __restrict__ g, double * __restrict__ uw,
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
void GINTgout2e_nabla1i(double * __restrict__ gout, double * __restrict__ g,
                        double ai, double aj) {

  int di = c_envs.stride_ijmax;
  int dj = c_envs.stride_ijmin;

  if (NROOTS < 7) {
    int nf = c_envs.nf;
    int16_t * idx = c_idx4c;
    int16_t * idy = idx + nf;
    int16_t * idz = idx + nf * 2;

    double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;

    int i, n, ix, iy, iz,
        ij_index_for_ix, i_index_for_ix, j_index_for_ix,
        ij_index_for_iy, i_index_for_iy, j_index_for_iy,
        ij_index_for_iz, i_index_for_iz, j_index_for_iz;

    for (i = 0; i < nf; i++) {

      s_ix = gout[i];
      s_iy = gout[i + nf];
      s_iz = gout[i + 2 * nf];
      s_jx = gout[i + 3 * nf];
      s_jy = gout[i + 4 * nf];
      s_jz = gout[i + 5 * nf];

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

#pragma unroll
      for (n = 0; n < NROOTS; ++n) {

        s_ix += -i_index_for_ix *
                g[ix + n - di] * g[iy + n] * g[iz + n]
                + 2.0 * ai * g[ix + n + di] * g[iy + n] * g[iz + n];
        s_iy += -i_index_for_iy *
                g[ix + n] * g[iy + n - di] * g[iz + n]
                + 2.0 * ai * g[ix + n] * g[iy + n + di] * g[iz + n];
        s_iz += -i_index_for_iz *
                g[ix + n] * g[iy + n] * g[iz + n - di]
                + 2.0 * ai * g[ix + n] * g[iy + n] * g[iz + n + di];
        s_jx += -j_index_for_ix *
                g[ix + n - dj] * g[iy + n] * g[iz + n]
                + 2.0 * aj * g[ix + n + dj] * g[iy + n] * g[iz + n];
        s_jy += -j_index_for_iy *
                g[ix + n] * g[iy + n - dj] * g[iz + n]
                + 2.0 * aj * g[ix + n] * g[iy + n + dj] * g[iz + n];
        s_jz += -j_index_for_iz *
                g[ix + n] * g[iy + n] * g[iz + n - dj]
                + 2.0 * aj * g[ix + n] * g[iy + n] * g[iz + n + dj];
      }

      gout[i] = s_ix;
      gout[i + nf] = s_iy;
      gout[i + 2 * nf] = s_iz;
      gout[i + 3 * nf] = s_jx;
      gout[i + 4 * nf] = s_jy;
      gout[i + 5 * nf] = s_jz;
    }
  } else {
    int nf = c_envs.nf;
    int16_t * idx = c_idx4c;
    if (nf > NFffff) {
      idx = c_envs.idx;
    }
    int16_t * idy = idx + nf;
    int16_t * idz = idx + nf * 2;
    double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;
    int i, n, ix, iy, iz,
        ij_index_for_ix, i_index_for_ix, j_index_for_ix,
        ij_index_for_iy, i_index_for_iy, j_index_for_iy,
        ij_index_for_iz, i_index_for_iz, j_index_for_iz;

    for (i = 0; i < nf; i++) {

      s_ix = gout[i];
      s_iy = gout[i + nf];
      s_iz = gout[i + 2 * nf];
      s_jx = gout[i + 3 * nf];
      s_jy = gout[i + 4 * nf];
      s_jz = gout[i + 5 * nf];

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


#pragma unroll
      for (n = 0; n < NROOTS; ++n) {
        s_ix += -i_index_for_ix *
                g[ix + n - di] * g[iy + n] * g[iz + n]
                + 2.0 * ai * g[ix + n + di] * g[iy + n] * g[iz + n];
        s_iy += -i_index_for_iy *
                g[ix + n] * g[iy + n - di] * g[iz + n]
                + 2.0 * ai * g[ix + n] * g[iy + n + di] * g[iz + n];
        s_iz += -i_index_for_iz *
                g[ix + n] * g[iy + n] * g[iz + n - di]
                + 2.0 * ai * g[ix + n] * g[iy + n] * g[iz + n + di];
        s_jx += -j_index_for_ix *
                g[ix + n - dj] * g[iy + n] * g[iz + n]
                + 2.0 * aj * g[ix + n + dj] * g[iy + n] * g[iz + n];
        s_jy += -j_index_for_iy *
                g[ix + n] * g[iy + n - dj] * g[iz + n]
                + 2.0 * aj * g[ix + n] * g[iy + n + dj] * g[iz + n];
        s_jz += -j_index_for_iz *
                g[ix + n] * g[iy + n] * g[iz + n - dj]
                + 2.0 * aj * g[ix + n] * g[iy + n] * g[iz + n + dj];
      }
      gout[i] = s_ix;
      gout[i + nf] = s_iy;
      gout[i + 2 * nf] = s_iz;
      gout[i + 3 * nf] = s_jx;
      gout[i + 4 * nf] = s_jy;
      gout[i + 5 * nf] = s_jz;
    }
  }
}

__device__
void GINTwrite_ints_s2_nabla1i(ERITensor eri, double * __restrict__ gout,
                               int ish, int jsh, int ksh, int lsh) {
  int * ao_loc = c_bpcache.ao_loc;
  size_t jstride = eri.stride_j;
  size_t kstride = eri.stride_k;
  size_t lstride = eri.stride_l;
  size_t xyz_stride = eri.n_elem;
  int i0 = ao_loc[ish];
  int i1 = ao_loc[ish + 1];
  int j0 = ao_loc[jsh];
  int j1 = ao_loc[jsh + 1];
  int k0 = ao_loc[ksh] - eri.ao_offsets_k;
  int k1 = ao_loc[ksh + 1] - eri.ao_offsets_k;
  int l0 = ao_loc[lsh] - eri.ao_offsets_l;
  int l1 = ao_loc[lsh + 1] - eri.ao_offsets_l;
  int nf = c_envs.nf;
  int i, j, k, l, n;
  double s_ix, s_iy, s_iz, s_jx, s_jy, s_jz;
  double * __restrict__ peri;
  double * __restrict__ peri_lk;
  for (n = 0, l = l0; l < l1; ++l) {
    for (k = k0; k < k1; ++k) {
      peri = eri.data + l * lstride + k * kstride;
      peri_lk = eri.data + k * lstride + l * kstride;
      for (j = j0; j < j1; ++j) {
        for (i = i0; i < i1; ++i, ++n) {
          s_ix = gout[n];
          s_iy = gout[n + nf];
          s_iz = gout[n + 2 * nf];
          s_jx = gout[n + 3 * nf];
          s_jy = gout[n + 4 * nf];
          s_jz = gout[n + 5 * nf];
          peri[i + jstride * j] = s_ix;
          peri[i + jstride * j + xyz_stride] = s_iy;
          peri[i + jstride * j + 2 * xyz_stride] = s_iz;
          peri[j + jstride * i] = s_jx;
          peri[j + jstride * i + xyz_stride] = s_jy;
          peri[j + jstride * i + 2 * xyz_stride] = s_jz;
          peri_lk[i + jstride * j] = s_ix;
          peri_lk[i + jstride * j + xyz_stride] = s_iy;
          peri_lk[i + jstride * j + 2 * xyz_stride] = s_iz;
          peri_lk[j + jstride * i] = s_jx;
          peri_lk[j + jstride * i + xyz_stride] = s_jy;
          peri_lk[j + jstride * i + 2 * xyz_stride] = s_jz;
        }
      }
    }
  }
}

template<int NROOTS, int GOUTSIZE>
__global__
static void
GINTfill_int2e_kernel_nabla1i(ERITensor eri, BasisProdOffsets offsets) {
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
  int nprim_j =
      c_bpcache.primitive_functions_offsets[jsh + 1]
      - c_bpcache.primitive_functions_offsets[jsh];

  double * exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];
  double * exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  int task_id = task_ij + ntasks_ij * task_kl;
  double * uw = c_envs.uw + task_id * nprim_ij * nprim_kl * NROOTS * 2;
  double gout[GOUTSIZE];
  double * g = gout + c_envs.nf * 6;
  int i;
  for (i = 0; i < c_envs.nf * 6; ++i) {
    gout[i] = 0;
  }

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
    for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
      GINTg0_2e_2d4d_nabla1i<NROOTS>(g, uw, norm, as_ish, as_jsh, as_ksh,
                                     as_lsh, ij, kl);

      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];

      GINTgout2e_nabla1i<NROOTS>(gout, g, ai, aj);
      uw += NROOTS * 2;
    }
  }

  GINTwrite_ints_s2_nabla1i(eri, gout, ish, jsh, ksh, lsh);
}

__global__
static void GINTfill_nabla1i_int2e_kernel0000(ERITensor eri,
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
  int * bas_pair2bra = c_bpcache.bas_pair2bra;
  int * bas_pair2ket = c_bpcache.bas_pair2ket;
  int ish = bas_pair2bra[bas_ij];
  int jsh = bas_pair2ket[bas_ij];
  int ksh = bas_pair2bra[bas_kl];
  int lsh = bas_pair2ket[bas_kl];

  int nprim_ij = c_envs.nprim_ij;
  int nprim_kl = c_envs.nprim_kl;
  int prim_ij = offsets.primitive_ij + task_ij * nprim_ij;
  int prim_kl = offsets.primitive_kl + task_kl * nprim_kl;
  double * __restrict__ a12 = c_bpcache.a12;
  double * __restrict__ e12 = c_bpcache.e12;
  double * __restrict__ x12 = c_bpcache.x12;
  double * __restrict__ y12 = c_bpcache.y12;
  double * __restrict__ z12 = c_bpcache.z12;
  int ij, kl;

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
    for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];
      double aij = a12[ij];
      double eij = e12[ij];
      double xij = x12[ij];
      double yij = y12[ij];
      double zij = z12[ij];
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

  int jstride = eri.stride_j;
  int kstride = eri.stride_k;
  int lstride = eri.stride_l;
  int * ao_loc = c_bpcache.ao_loc;
  int i0 = ao_loc[ish];
  int j0 = ao_loc[jsh];
  int k0 = ao_loc[ksh] - eri.ao_offsets_k;
  int l0 = ao_loc[lsh] - eri.ao_offsets_l;
  double * __restrict__ eri_ij =
      eri.data + l0 * lstride + k0 * kstride + j0 * jstride + i0;
  double * __restrict__ eri_ji =
      eri.data + l0 * lstride + k0 * kstride + i0 * jstride + j0;
  double * __restrict__ eri_ij_lk =
      eri.data + k0 * lstride + l0 * kstride + j0 * jstride + i0;
  double * __restrict__ eri_ji_lk =
      eri.data + k0 * lstride + l0 * kstride + i0 * jstride + j0;

  size_t xyz_stride = eri.n_elem;

  eri_ij[0] = gout0;
  eri_ij[1 * xyz_stride] = gout1;
  eri_ij[2 * xyz_stride] = gout2;
  eri_ij_lk[0] = gout0;
  eri_ij_lk[1 * xyz_stride] = gout1;
  eri_ij_lk[2 * xyz_stride] = gout2;

  eri_ji[0] = gout0_prime;
  eri_ji[1 * xyz_stride] = gout1_prime;
  eri_ji[2 * xyz_stride] = gout2_prime;
  eri_ji_lk[0] = gout0_prime;
  eri_ji_lk[1 * xyz_stride] = gout1_prime;
  eri_ji_lk[2 * xyz_stride] = gout2_prime;
}


#if POLYFIT_ORDER >= 4

template<>
__global__
void GINTfill_int2e_kernel_nabla1i<4, NABLAGOUTSIZE4>(ERITensor eri,
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
  int nprim_j =
      c_bpcache.primitive_functions_offsets[jsh + 1]
      - c_bpcache.primitive_functions_offsets[jsh];

  double * exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];
  double * exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  double uw[8];
  double gout[NABLAGOUTSIZE4];
  double * g = gout + c_envs.nf * 6;
  int i;
  for (i = 0; i < 6 * c_envs.nf; ++i) {
    gout[i] = 0;
  }

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
    for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
      double aij = a12[ij];
      double xij = x12[ij];
      double yij = y12[ij];
      double zij = z12[ij];
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
      GINTg0_2e_2d4d_nabla1i<4>(g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij,
                                kl);
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];

      GINTgout2e_nabla1i<4>(gout, g, ai, aj);
    }
  }

  GINTwrite_ints_s2_nabla1i(eri, gout, ish, jsh, ksh, lsh);
}

#endif

#if POLYFIT_ORDER >= 5

template<>
__global__
void GINTfill_int2e_kernel_nabla1i<5, NABLAGOUTSIZE5>(ERITensor eri,
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
  int nprim_j =
      c_bpcache.primitive_functions_offsets[jsh + 1]
      - c_bpcache.primitive_functions_offsets[jsh];

  double * exponent_i =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[ish];
  double * exponent_j =
      c_bpcache.exponents + c_bpcache.primitive_functions_offsets[jsh];

  double uw[10];
  double gout[NABLAGOUTSIZE5];
  double * g = gout + c_envs.nf * 6;
  int i;
  for (i = 0; i < 6 * c_envs.nf; ++i) {
    gout[i] = 0;
  }

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
    for (kl = prim_kl; kl < prim_kl + nprim_kl; ++kl) {
      double aij = a12[ij];
      double xij = x12[ij];
      double yij = y12[ij];
      double zij = z12[ij];
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
      GINTg0_2e_2d4d_nabla1i<5>(g, uw, norm, as_ish, as_jsh, as_ksh, as_lsh, ij,
                                kl);
      double ai = exponent_i[(ij - prim_ij) / nprim_j];
      double aj = exponent_j[(ij - prim_ij) % nprim_j];

      GINTgout2e_nabla1i<5>(gout, g, ai, aj);
    }
  }

  GINTwrite_ints_s2_nabla1i(eri, gout, ish, jsh, ksh, lsh);
}

#endif
