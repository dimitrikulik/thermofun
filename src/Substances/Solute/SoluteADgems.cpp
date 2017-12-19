#include "Substances/Solute/SoluteADgems.h"
#include "Common/Exception.h"
#include "Substance.h"
#include "ThermoProperties.h"
#include "ThermoParameters.h"

namespace ThermoFun {

auto thermoPropertiesAqSoluteAD(Reaktoro_::Temperature T, Reaktoro_::Pressure P, Substance subst, ThermoPropertiesSubstance tps, const ThermoPropertiesSubstance& wtp,
                                const ThermoPropertiesSubstance& wigp, const PropertiesSolvent& wp, const ThermoPropertiesSubstance& wtpr, const ThermoPropertiesSubstance& wigpr, const PropertiesSolvent& wpr) -> ThermoPropertiesSubstance
{
//    Reaktoro::ThermoScalar CaltoJ(cal_to_J);
    // calculate infinite dilution properties of aqueous species at T and P of interest
//    Reaktoro_::ThermoScalar rho, alp, bet, dalpT, dH0k;
////    Reaktoro::ThermoScalar R_CONST (R_CONSTANT);
//    Reaktoro_::ThermoScalar Gig, Sig, CPig, Gw, Sw, CPw;
    Reaktoro_::ThermoScalar Geos, Veos, Seos, CPeos, Heos;
    Reaktoro_::ThermoScalar Gids, /*Vids,*/ Sids, CPids, Hids;
    Reaktoro_::ThermoScalar Geos298, Veos298, Seos298, CPeos298, Heos298;

    Reaktoro_::Temperature Tk = T;
    auto Tr (298.15);
    Reaktoro_::Pressure Pbar = P;
    auto Pr = 1.0;

    ThermoPropertiesSubstance state = tps;

    const auto ADparam = subst.thermoParameters().Cp_nonElectrolyte_coeff;

    if (ADparam.size() == 0)
    {
        Exception exception;
        exception.error << "Error in Akinfiev Diamond EOS";
        exception.reason << "There are no model parameters given for "<<subst.symbol() << ".";
        exception.line = __LINE__;
        RaiseError(exception);
    }

    auto dH0k = (-182161.88);  // enthapy of ideal gas water at 0 K

    // Properties of water at Tr,Pr (25 deg C, 1 bar) from SUPCRT92 routines
    // adopted H2O ideal gas data from NIST-TRC database
    auto Gig = wigpr.gibbs_energy;
    auto Sig = wigpr.entropy;
    auto CPig = wigpr.heat_capacity_cp;

    auto Gw    = wtpr.gibbs_energy;
    auto Sw    = wtpr.entropy;
    auto CPw   = wtpr.heat_capacity_cp;
    auto rho   = wpr.density/1000;
    auto alp   = wpr.Alpha;
    auto bet   = wpr.Beta*1e05;
    auto dalpT = wpr.dAldT;

//    Gig = -228526.66;
//    Sig = 188.72683;
//    CPig = 33.58743;

//    Gw = -237181.38;
//    Sw = 69.92418;
//    CPw = 75.36053;
//    rho = 0.99706136;
//    alp = 2.59426542e-4;
//    bet = 4.52187717e-5;
//    dalpT = 9.56485765e-6;

    Akinfiev_EOS_increments(Tr, Pr, Gig, Sig, CPig, Gw, Sw, CPw, rho, alp, bet, dalpT, ADparam,
                       Geos298, Veos298, Seos298, CPeos298, Heos298 );

    // Getting back ideal gas properties corrected for T of interest
    // by substracting properties of hydration at Tr, Pr
    Gids = state.gibbs_energy -= Geos298;
        // Vids = aW.twp->V -= Veos298;
    Sids = state.entropy -= Seos298;
        // CPids = aW.twp->Cp -= CPeos298;
    CPids = state.heat_capacity_cp;
    Hids = state.enthalpy -= Heos298;

    // Properties of water at T,P of interest, modified 06.02.2008 (TW)
//	Tk = aW.twp->T;
//	Pbar = aW.twp->P;
    Gig = wigp.gibbs_energy; // aWp.Gigw[aSpc.isat]*R_CONST*Tk + dH0k;  // converting normalized ideal gas values
    Sig = wigp.entropy; //aWp.Sigw[aSpc.isat]*R_CONST;
    CPig = wigp.heat_capacity_cp; // aWp.Cpigw[aSpc.isat]*R_CONST;
    Gw = wtp.gibbs_energy; //aWp.Gw[aSpc.isat]*CaltoJ;
    Sw = wtp.entropy; //aWp.Sw[aSpc.isat]*CaltoJ;
    CPw = wtp.heat_capacity_cp; //aWp.Cpw[aSpc.isat]*CaltoJ;
    /// units????
    rho = wp.density / 1000; // aSta.Dens[aSpc.isat];
    alp = wp.Alpha; // aWp.Alphaw[aSpc.isat];
    bet = wp.Beta*1e05; // aWp.Betaw[aSpc.isat];
    dalpT = wp.dAldT; // aWp.dAldT[aSpc.isat];

    Akinfiev_EOS_increments(Tk, Pbar, Gig, Sig, CPig, Gw, Sw, CPw, rho, alp, bet, dalpT, ADparam,
                       Geos, Veos, Seos, CPeos, Heos );

    // Getting dissolved gas properties corrected for T,P of interest
    // by adding properties of hydration at T,P
    Gids.ddp = 0.0; Sids.ddp = 0.0; CPids.ddp = 0.0;
    state.gibbs_energy     = Gids + Geos + Seos298.val*(Tk.val-Tr);  // S(T-Tr) corrected for dSh at Tr,Pr
    // aW.twp->V = Vids + Veos;
    state.volume           = Veos;
    state.entropy          = Sids + Seos;
    state.heat_capacity_cp = CPids + CPeos;
    state.enthalpy         = Hids + Heos;
    state.internal_energy  = state.enthalpy - Pbar*state.volume;
    state.helmholtz_energy = state.internal_energy - Tk*state.entropy;

    subst.checkCalcMethodBounds("Akinfiev and Diamond model", Tk.val-C_to_K, Pbar.val, tps);

    return state;
}

void Akinfiev_EOS_increments(Reaktoro_::Temperature Tk, Reaktoro_::Pressure P, Reaktoro_::ThermoScalar Gig, Reaktoro_::ThermoScalar Sig, Reaktoro_::ThermoScalar CPig,
        Reaktoro_::ThermoScalar Gw, Reaktoro_::ThermoScalar Sw, Reaktoro_::ThermoScalar CPw, Reaktoro_::ThermoScalar rho, Reaktoro_::ThermoScalar alp, Reaktoro_::ThermoScalar bet, Reaktoro_::ThermoScalar dalpT, vd ADparam,
        Reaktoro_::ThermoScalar &Geos, Reaktoro_::ThermoScalar &Veos, Reaktoro_::ThermoScalar &Seos, Reaktoro_::ThermoScalar &CPeos, Reaktoro_::ThermoScalar &Heos )
{

    Reaktoro_::ThermoScalar derP, derT, der2T;
    Reaktoro_::ThermoScalar deltaB, lnKH, Nw, xi, aa, bb, RT;
    Reaktoro_::ThermoScalar fug, vol, drhoT, drhoP, d2rhoT, lnfug, Gres, Sres, CPres;
    const Reaktoro_::ThermoScalar RR (83.1451); const Reaktoro_::ThermoScalar R_CONST (R_CONSTANT);
    const Reaktoro_::ThermoScalar MW (18.01528);

    RT = Tk*R_CONST;
    // EOS coefficients
    xi = ADparam[0];
    aa = ADparam[1];
    bb = ADparam[2];

    Gres = Gw-Gig;
    Sres = Sw-Sig;
    CPres = CPw - CPig;
    lnfug = Gres/RT;
    fug = exp(lnfug);
    vol = MW/rho;
    drhoT = - alp*rho;
    drhoP = bet*rho;
    d2rhoT = rho*(pow(alp,2.)-dalpT);

    // calculation of infinite dilution properties
    Nw = 1000./MW;
    deltaB = 0.5*(aa + bb*pow((1000./Tk),0.5));
    lnKH = (1.-xi)*log(fug) + xi*log(RR*Tk/MW*rho) + 2.*rho*deltaB;

    Geos = - R_CONST*Tk*log(Nw) + (1.-xi)*R_CONST*Tk*log(fug) + R_CONST*Tk*xi*log(RR*Tk/MW*rho)
                + R_CONST*Tk*rho*(aa+bb*pow((1000./Tk),0.5));
    derP = aa*Tk*drhoP + bb*pow(10.,1.5)*pow(Tk,0.5)*drhoP;
    derT = aa*(rho+Tk*drhoT) + bb*(0.5*pow(10.,1.5)*pow(Tk,-0.5)*rho + pow(10.,1.5)*pow(Tk,0.5)*drhoT);
    der2T = aa*(2.*drhoT+Tk*d2rhoT) + bb*((-0.25)*pow(10.,1.5)*pow(Tk,-1.5)*rho
                + pow(10.,1.5)*pow(Tk,-0.5)*drhoT + pow(10.,1.5)*pow(Tk,0.5)*d2rhoT);

    Veos = (vol*(1.-xi) + xi*RR*Tk*(1./rho)*drhoP + RR*derP)/10.;
    Seos = (1.-xi)*(Sres) + R_CONST*log(Nw) - R_CONST*(xi + xi*log(RR*Tk/MW) + xi*log(rho)
                + xi*Tk*(1./rho)*drhoT) - R_CONST*derT;
    CPeos = (1.-xi)*(CPres) - R_CONST*(xi + 2.*xi*Tk*(1./rho)*drhoT
                - xi*pow(Tk,2.)/pow(rho,2.)*pow(drhoT,2.) + xi*pow(Tk,2.)/rho*d2rhoT) - R_CONST*Tk*der2T;
    Heos = Geos + Tk*Seos;
}

}