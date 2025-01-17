from nasu import calc, proc
import numpy as np

# ==================================================================================================
class tR():
    def __init__(self, t_s, R_m, r_m, rho, d, e=None):
        self.t_s = t_s
        self.R_m = R_m
        self.r_m = r_m
        self.rho = rho
        self.d = d
        self.e = e

    def rm_noisy_time(self):
        _rer = self.e / self.d
        _rer_med = np.nanmedian(_rer, axis=1)
        _datlist = [self.r_m, self.rho, self.d, self.e]
        _idx = np.where(_rer_med < 1.)[0]
        self.t_s = self.t_s[_idx]
        _datlist = proc.get_dats_at_idxs(_datlist, _idx, axis=0)
        self.r_m, self.rho, self.d, self.e = _datlist

    def rm_noisy_ch(self):
        _rer = self.e / self.d
        _rer_med = np.nanmedian(_rer, axis=0)
        _datlist = [self.r_m, self.rho, self.d, self.e]
        _idx = np.where(_rer_med < 1.)[0]
        self.R_m = self.R_m[_idx]
        _datlist = proc.get_dats_at_idxs(_datlist, _idx, axis=1)
        self.r_m, self.rho, self.d, self.e = _datlist

    def cut_by_rho(self, rho_cut):
        self.rho_cut = rho_cut
        self.rho[np.abs(self.rho) > self.rho_cut] = np.nan
        _Ridxs = ~np.isnan(self.rho).any(axis=0)
        self.R_m = self.R_m[_Ridxs]
        _datlist = [self.r_m, self.rho, self.d, self.e]
        _datlist = proc.get_dats_at_idxs(_datlist, _Ridxs, axis=1)
        self.r_m, self.rho, self.d, self.e = _datlist

    def average(self, axis=None, skipnan=True):
        _avg, _std, _ste = calc.average(dat=self.d, err=self.e, axis=axis, skipnan=skipnan)
        self.avg = calc.struct()
        self.avg.d = _avg
        self.avg.s = _std
        self.avg.e = _ste
        return self.avg
    
    def polyfit(self, polyN=10):

        # a. 2)
        _o = calc.polyN_LSM_der(xx=self.r_m, yy=self.d, polyN=polyN, yErr=self.e, parity="even")
        self.pfit = tR(self.t_s, self.R_m, self.r_m, self.rho, _o.yHut, _o.yHutErr)
        self.pfit.grad = tR(self.t_s, self.R_m, self.r_m, self.rho, _o.yHutDer, _o.yHutDerErr)

        return self.pfit

    def gradient(self):

        # b. 1)
        _drdR = np.gradient(self.r_m, self.R_m, edge_order=2, axis=-1)
        _dddR = np.gradient(self.d, self.R_m, edge_order=2, axis=-1)
        _dddR_err = np.abs(np.gradient(self.e, self.R_m, edge_order=2, axis=-1))
        _dddr, _dddr_err = calc.dMdreff(_dddR, _drdR, _dddR_err)

        self.grad = tR(self.t_s, self.R_m, self.r_m, self.rho, _dddr, _dddr_err)
        
        return self.grad
    
    def Lscale(self, Rax):
        self.Rax = Rax
        _L, _L_err, _RL, _RL_err \
            = calc.Lscale(self.d, self.grad.d, self.Rax, self.e, self.grad.e)
        self.L = tR(self.t_s, self.R_m, self.r_m, self.rho, _L, _L_err)
        self.RL = tR(self.t_s, self.R_m, self.r_m, self.rho, _RL, _RL_err)
        return self.L, self.RL
    
    def t_at(self, time):
        _, _datlist_at = proc.getTimeIdxAndDats(self.t_s, time, [self.t_s, self.r_m, self.rho, self.d, self.e])
        self.tat = tR(t_s=_datlist_at[0], R_m=self.R_m, r_m=_datlist_at[1], rho=_datlist_at[2], d=_datlist_at[3], e=_datlist_at[4])
        return self.tat
    
    def R_at(self, Rat):
        _, _R, _datlist = proc.get_idx_and_dats(coord=self.R, coord_at=Rat, dat_list=[self.r_m, self.rho, self.d, self.e], axis=1)
        self.Rat = tR(t_s=self.t_s, R_m=_R, r_m=_datlist[0], rho=_datlist[1], d=_datlist[2], e=_datlist[3])

    def t_window(self, tstart, tend, include_outerside=True):
        _, _t, _datlist = proc.get_idxs_and_dats(coord=self.t_s, coord_lim=(tstart, tend), dat_list=[self.r_m, self.rho, self.d, self.e], 
                                                 axis=0, include_outerside=include_outerside)
        self.twin = tR(t_s=_t, R_m=self.R_m, r_m=_datlist[0], rho=_datlist[1], d=_datlist[2], e=_datlist[3])
        self.twin.tstart = tstart
        self.twin.tend = tend
        return self.twin
    
    def R_window(self, Rat, dR=0.106, include_outerside=True):
        Rin = Rat - 0.5 * dR
        Rout = Rat + 0.5 * dR
        _, _R, _datlist = proc.get_idxs_and_dats(coord=self.R_m, coord_lim=(Rin, Rout), dat_list=[self.r_m, self.rho, self.d, self.e], 
                                                 axis=1, include_outerside=include_outerside)
        self.Rwin = tR(t_s=self.t_s, R_m=_R, r_m=_datlist[0], rho=_datlist[1], d=_datlist[2], e=_datlist[3])
        self.Rwin.Rat = Rat
        self.Rwin.dR = dR
        self.Rwin.Rin = Rin
        self.Rwin.Rout = Rout
        return self.Rwin

    # Te, or ne_calFIR 分布データの処理
    # ※ 勾配計算は, 物理量をMとして, dM/dR と　dreff/dRをそれぞれ計算し、(dM/dR) / (dreff/dR) で dM/dreffを計算するのが良い。
    #    （データ仕様の都合上）
    # a. 生データのまま
    #   1) 局所直線フィッティングで勾配をそのまま計算する。スケール長計算時のTe or neの値には0次項を使う。
    #   ok! 2) 全点n次多項式フィッティングを行う（横軸はreffで、左右対称な関数）。導関数から勾配を計算する。
    # b. Te_fit or ne_fit を使う。
    #   ok! 1) 2次中心差分で勾配を計算する。
    #   2) 5点ステンシル中心差分で勾配を計算する。
    # c. 平均処理（時間平均(ok!)・ショット平均・空間平均（移動平均）などで処理）してばらつきを小さくしたデータを使う。
    #   1) 局所直線フィッティングで勾配を計算する。スケール長計算時のTe or neの値には0次項を使う。
    #   2) 全点n次多項式フィッティングを行う。導関数から勾配を計算する。
    # c.はオプションとして実装しておく。平均処理した後に行えるように。
