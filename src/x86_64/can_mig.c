/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mech_api.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__can
#define _nrn_initial _nrn_initial__can
#define nrn_cur _nrn_cur__can
#define _nrn_current _nrn_current__can
#define nrn_jacob _nrn_jacob__can
#define nrn_state _nrn_state__can
#define _net_receive _net_receive__can 
#define rates rates__can 
#define states states__can 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gcanbar _p[0]
#define gcanbar_columnindex 0
#define ica _p[1]
#define ica_columnindex 1
#define gcan _p[2]
#define gcan_columnindex 2
#define minf _p[3]
#define minf_columnindex 3
#define hinf _p[4]
#define hinf_columnindex 4
#define taum _p[5]
#define taum_columnindex 5
#define tauh _p[6]
#define tauh_columnindex 6
#define m _p[7]
#define m_columnindex 7
#define h _p[8]
#define h_columnindex 8
#define cai _p[9]
#define cai_columnindex 9
#define cao _p[10]
#define cao_columnindex 10
#define Dm _p[11]
#define Dm_columnindex 11
#define Dh _p[12]
#define Dh_columnindex 12
#define v _p[13]
#define v_columnindex 13
#define _g _p[14]
#define _g_columnindex 14
#define _ion_cai	*_ppvar[0]._pval
#define _ion_cao	*_ppvar[1]._pval
#define _ion_ica	*_ppvar[2]._pval
#define _ion_dicadv	*_ppvar[3]._pval
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 extern double celsius;
 /* declaration of user functions */
 static void _hoc_KTF(void);
 static void _hoc_alpmt(void);
 static void _hoc_alpm(void);
 static void _hoc_alph(void);
 static void _hoc_betmt(void);
 static void _hoc_betm(void);
 static void _hoc_beth(void);
 static void _hoc_efun(void);
 static void _hoc_ghk(void);
 static void _hoc_h2(void);
 static void _hoc_rates(void);
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_can", _hoc_setdata,
 "KTF_can", _hoc_KTF,
 "alpmt_can", _hoc_alpmt,
 "alpm_can", _hoc_alpm,
 "alph_can", _hoc_alph,
 "betmt_can", _hoc_betmt,
 "betm_can", _hoc_betm,
 "beth_can", _hoc_beth,
 "efun_can", _hoc_efun,
 "ghk_can", _hoc_ghk,
 "h2_can", _hoc_h2,
 "rates_can", _hoc_rates,
 0, 0
};
#define KTF KTF_can
#define alpmt alpmt_can
#define alpm alpm_can
#define alph alph_can
#define betmt betmt_can
#define betm betm_can
#define beth beth_can
#define efun efun_can
#define ghk ghk_can
#define h2 h2_can
 extern double KTF( _threadargsprotocomma_ double );
 extern double alpmt( _threadargsprotocomma_ double );
 extern double alpm( _threadargsprotocomma_ double );
 extern double alph( _threadargsprotocomma_ double );
 extern double betmt( _threadargsprotocomma_ double );
 extern double betm( _threadargsprotocomma_ double );
 extern double beth( _threadargsprotocomma_ double );
 extern double efun( _threadargsprotocomma_ double );
 extern double ghk( _threadargsprotocomma_ double , double , double );
 extern double h2( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define USEGHK USEGHK_can
 double USEGHK = 1;
#define a0m a0m_can
 double a0m = 0.03;
#define erev erev_can
 double erev = 100;
#define gmm gmm_can
 double gmm = 0.1;
#define hmin hmin_can
 double hmin = 3;
#define ki ki_can
 double ki = 0.001;
#define mmin mmin_can
 double mmin = 0.2;
#define q10 q10_can
 double q10 = 5;
#define vhalfm vhalfm_can
 double vhalfm = -14;
#define zetam zetam_can
 double zetam = 2;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "ki_can", "mM",
 "gcanbar_can", "mho/cm2",
 "ica_can", "mA/cm2",
 "gcan_can", "mho/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "ki_can", &ki_can,
 "q10_can", &q10_can,
 "mmin_can", &mmin_can,
 "hmin_can", &hmin_can,
 "a0m_can", &a0m_can,
 "zetam_can", &zetam_can,
 "vhalfm_can", &vhalfm_can,
 "gmm_can", &gmm_can,
 "USEGHK_can", &USEGHK_can,
 "erev_can", &erev_can,
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(NrnThread*, _Memb_list*, int);
static void nrn_state(NrnThread*, _Memb_list*, int);
 static void nrn_cur(NrnThread*, _Memb_list*, int);
static void  nrn_jacob(NrnThread*, _Memb_list*, int);
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[4]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"can",
 "gcanbar_can",
 0,
 "ica_can",
 "gcan_can",
 "minf_can",
 "hinf_can",
 "taum_can",
 "tauh_can",
 0,
 "m_can",
 "h_can",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 15, _prop);
 	/*initialize range parameters*/
 	gcanbar = 0.0003;
 	_prop->param = _p;
 	_prop->param_size = 15;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 5, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[0]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[1]._pval = &prop_ion->param[2]; /* cao */
 	_ppvar[2]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[3]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _can_mig_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 15, 5);
  hoc_register_dparam_semantics(_mechtype, 0, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 can /mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/can_mig.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 96520.0;
 static double R = 8.3134;
 static double KTOMV = .0853;
static int _reset;
static char *modelname = "n-calcium channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
double h2 ( _threadargsprotocomma_ double _lcai ) {
   double _lh2;
 _lh2 = ki / ( ki + _lcai ) ;
   
return _lh2;
 }
 
static void _hoc_h2(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  h2 ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double ghk ( _threadargsprotocomma_ double _lv , double _lci , double _lco ) {
   double _lghk;
 double _lnu , _lf ;
 _lf = KTF ( _threadargscomma_ celsius ) / 2.0 ;
   _lnu = _lv / _lf ;
   _lghk = - _lf * ( 1. - ( _lci / _lco ) * exp ( _lnu ) ) * efun ( _threadargscomma_ _lnu ) ;
   
return _lghk;
 }
 
static void _hoc_ghk(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  ghk ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) );
 hoc_retpushx(_r);
}
 
double KTF ( _threadargsprotocomma_ double _lcelsius ) {
   double _lKTF;
 _lKTF = ( ( 25. / 293.15 ) * ( _lcelsius + 273.15 ) ) ;
   
return _lKTF;
 }
 
static void _hoc_KTF(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  KTF ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double efun ( _threadargsprotocomma_ double _lz ) {
   double _lefun;
 if ( fabs ( _lz ) < 1e-4 ) {
     _lefun = 1.0 - _lz / 2.0 ;
     }
   else {
     _lefun = _lz / ( exp ( _lz ) - 1.0 ) ;
     }
   
return _lefun;
 }
 
static void _hoc_efun(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  efun ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double alph ( _threadargsprotocomma_ double _lv ) {
   double _lalph;
 _lalph = 1.6e-4 * exp ( - _lv / 48.4 ) ;
   
return _lalph;
 }
 
static void _hoc_alph(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alph ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double beth ( _threadargsprotocomma_ double _lv ) {
   double _lbeth;
 _lbeth = 1.0 / ( exp ( ( - _lv + 39.0 ) / 10. ) + 1. ) ;
   
return _lbeth;
 }
 
static void _hoc_beth(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  beth ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double alpm ( _threadargsprotocomma_ double _lv ) {
   double _lalpm;
 _lalpm = 0.1967 * ( - 1.0 * _lv + 19.88 ) / ( exp ( ( - 1.0 * _lv + 19.88 ) / 10.0 ) - 1.0 ) ;
   
return _lalpm;
 }
 
static void _hoc_alpm(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alpm ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double betm ( _threadargsprotocomma_ double _lv ) {
   double _lbetm;
 _lbetm = 0.046 * exp ( - _lv / 20.73 ) ;
   
return _lbetm;
 }
 
static void _hoc_betm(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betm ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double alpmt ( _threadargsprotocomma_ double _lv ) {
   double _lalpmt;
 _lalpmt = exp ( 0.0378 * zetam * ( _lv - vhalfm ) ) ;
   
return _lalpmt;
 }
 
static void _hoc_alpmt(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alpmt ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double betmt ( _threadargsprotocomma_ double _lv ) {
   double _lbetmt;
 _lbetmt = exp ( 0.0378 * zetam * gmm * ( _lv - vhalfm ) ) ;
   
return _lbetmt;
 }
 
static void _hoc_betmt(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  betmt ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / taum ;
   Dh = ( hinf - h ) / tauh ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taum )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / tauh )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taum)))*(- ( ( ( minf ) ) / taum ) / ( ( ( ( - 1.0 ) ) ) / taum ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / tauh)))*(- ( ( ( hinf ) ) / tauh ) / ( ( ( ( - 1.0 ) ) ) / tauh ) - h) ;
   }
  return 0;
}
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   double _la , _lb , _lqt ;
 _lqt = pow( q10 , ( ( celsius - 25.0 ) / 10.0 ) ) ;
   _la = alpm ( _threadargscomma_ _lv ) ;
   _lb = 1.0 / ( _la + betm ( _threadargscomma_ _lv ) ) ;
   minf = _la * _lb ;
   taum = betmt ( _threadargscomma_ _lv ) / ( _lqt * a0m * ( 1.0 + alpmt ( _threadargscomma_ _lv ) ) ) ;
   if ( taum < mmin / _lqt ) {
     taum = mmin / _lqt ;
     }
   _la = alph ( _threadargscomma_ _lv ) ;
   _lb = 1.0 / ( _la + beth ( _threadargscomma_ _lv ) ) ;
   hinf = _la * _lb ;
   tauh = 80.0 ;
   if ( tauh < hmin ) {
     tauh = hmin ;
     }
    return 0; }
 
static void _hoc_rates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rates ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int _ode_count(int _type){ return 2;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
     _ode_spec1 (_p, _ppvar, _thread, _nt);
  }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 2; ++_i) {
		_pv[_i] = _pp + _slist1[_i];  _pvdot[_i] = _pp + _dlist1[_i];
		_cvode_abstol(_atollist, _atol, _i);
	}
 }
 
static void _ode_matsol_instance1(_threadargsproto_) {
 _ode_matsol1 (_p, _ppvar, _thread, _nt);
 }
 
static void _ode_matsol(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
  cai = _ion_cai;
  cao = _ion_cao;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_ca_sym, _ppvar, 0, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 1, 2);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
 {
   rates ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   }
 
}
}

static void nrn_init(NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
  cai = _ion_cai;
  cao = _ion_cao;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gcan = gcanbar * m * m * h * h2 ( _threadargscomma_ cai ) ;
   if ( USEGHK  == 1.0 ) {
     ica = gcan * ghk ( _threadargscomma_ v , cai , cao ) ;
     }
   else {
     ica = gcan * ( v - erev ) ;
     }
   }
 _current += ica;

} return _current;
}

static void nrn_cur(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
  cai = _ion_cai;
  cao = _ion_cao;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dica;
  _dica = ica;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dicadv += (_dica - ica)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica ;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v = 0.0; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
 _nd = _ml->_nodelist[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v=_v;
{
  cai = _ion_cai;
  cao = _ion_cao;
 {   states(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/can_mig.mod";
static const char* nmodl_file_text = 
  "TITLE n-calcium channel\n"
  ": n-type calcium channel\n"
  ": MODELDB 126814 CA3 by Safiulina et al - http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=126814\n"
  ": by Michele Migliore\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "\n"
  "	FARADAY = 96520 (coul)\n"
  "	R = 8.3134 (joule/degC)\n"
  "	KTOMV = .0853 (mV/degC)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	v (mV)\n"
  "	celsius 		(degC)\n"
  "	gcanbar=.0003 (mho/cm2)\n"
  "	ki=.001 (mM)\n"
  "	cai=50.e-6 (mM)\n"
  "	cao = 2  (mM)\n"
  "	q10=5\n"
  "	mmin = 0.2\n"
  "	hmin = 3\n"
  "	a0m =0.03\n"
  "	zetam = 2\n"
  "	vhalfm = -14\n"
  "	gmm=0.1	\n"
  "        USEGHK=1\n"
  "        erev = 100\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX can\n"
  "	USEION ca READ cai,cao WRITE ica\n"
  "        RANGE gcanbar, ica, gcan       \n"
  "        RANGE hinf,minf,taum,tauh\n"
  "        GLOBAL USEGHK\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	m h \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	ica (mA/cm2)\n"
  "        gcan  (mho/cm2) \n"
  "        minf\n"
  "        hinf\n"
  "        taum\n"
  "        tauh\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "        rates(v)\n"
  "        m = minf\n"
  "        h = hinf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	gcan = gcanbar*m*m*h*h2(cai)\n"
  "        if (USEGHK == 1) {\n"
  "          ica = gcan*ghk(v,cai,cao)\n"
  "        } else {\n"
  "          ica = gcan*(v-erev)\n"
  "        }\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "FUNCTION h2(cai(mM)) {\n"
  "	h2 = ki/(ki+cai)\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION ghk(v(mV), ci(mM), co(mM)) (mV) {\n"
  "        LOCAL nu,f\n"
  "\n"
  "        f = KTF(celsius)/2\n"
  "        nu = v/f\n"
  "        ghk=-f*(1. - (ci/co)*exp(nu))*efun(nu)\n"
  "}\n"
  "\n"
  "FUNCTION KTF(celsius (degC)) (mV) {\n"
  "        KTF = ((25./293.15)*(celsius + 273.15))\n"
  "}\n"
  "\n"
  "\n"
  "FUNCTION efun(z) {\n"
  "	if (fabs(z) < 1e-4) {\n"
  "		efun = 1 - z/2\n"
  "	}else{\n"
  "		efun = z/(exp(z) - 1)\n"
  "	}\n"
  "}\n"
  "\n"
  "FUNCTION alph(v(mV)) {\n"
  "	alph = 1.6e-4*exp(-v/48.4)\n"
  "}\n"
  "\n"
  "FUNCTION beth(v(mV)) {\n"
  "	beth = 1/(exp((-v+39.0)/10.)+1.)\n"
  "}\n"
  "\n"
  "FUNCTION alpm(v(mV)) {\n"
  "	alpm = 0.1967*(-1.0*v+19.88)/(exp((-1.0*v+19.88)/10.0)-1.0)\n"
  "}\n"
  "\n"
  "FUNCTION betm(v(mV)) {\n"
  "	betm = 0.046*exp(-v/20.73)\n"
  "}\n"
  "\n"
  "FUNCTION alpmt(v(mV)) {\n"
  "  alpmt = exp(0.0378*zetam*(v-vhalfm)) \n"
  "}\n"
  "\n"
  "FUNCTION betmt(v(mV)) {\n"
  "  betmt = exp(0.0378*zetam*gmm*(v-vhalfm)) \n"
  "}\n"
  "\n"
  "UNITSON\n"
  "\n"
  "DERIVATIVE states {     : exact when v held constant; integrates over dt step\n"
  "        rates(v)\n"
  "        m' = (minf - m)/taum\n"
  "        h' = (hinf - h)/tauh\n"
  "}\n"
  "\n"
  "PROCEDURE rates(v (mV)) { :callable from hoc\n"
  "        LOCAL a, b, qt\n"
  "        qt=q10^((celsius-25)/10)\n"
  "        a = alpm(v)\n"
  "        b = 1/(a + betm(v))\n"
  "        minf = a*b\n"
  "	taum = betmt(v)/(qt*a0m*(1+alpmt(v)))\n"
  "	if (taum<mmin/qt) {taum=mmin/qt}\n"
  "        a = alph(v)\n"
  "        b = 1/(a + beth(v))\n"
  "        hinf = a*b\n"
  ":	tauh=b/qt\n"
  "	tauh= 80\n"
  "	if (tauh<hmin) {tauh=hmin}\n"
  "}\n"
  ;
#endif
