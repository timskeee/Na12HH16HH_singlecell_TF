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
 
#define nrn_init _nrn_init__cat
#define _nrn_initial _nrn_initial__cat
#define nrn_cur _nrn_cur__cat
#define _nrn_current _nrn_current__cat
#define nrn_jacob _nrn_jacob__cat
#define nrn_state _nrn_state__cat
#define _net_receive _net_receive__cat 
#define rates rates__cat 
#define states states__cat 
 
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
#define gcatbar _p[0]
#define gcatbar_columnindex 0
#define ica _p[1]
#define ica_columnindex 1
#define gcat _p[2]
#define gcat_columnindex 2
#define hinf _p[3]
#define hinf_columnindex 3
#define htau _p[4]
#define htau_columnindex 4
#define minf _p[5]
#define minf_columnindex 5
#define mtau _p[6]
#define mtau_columnindex 6
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
 static void _hoc_alph(void);
 static void _hoc_betmt(void);
 static void _hoc_beth(void);
 static void _hoc_efun(void);
 static void _hoc_ghk(void);
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
 "setdata_cat", _hoc_setdata,
 "KTF_cat", _hoc_KTF,
 "alpmt_cat", _hoc_alpmt,
 "alph_cat", _hoc_alph,
 "betmt_cat", _hoc_betmt,
 "beth_cat", _hoc_beth,
 "efun_cat", _hoc_efun,
 "ghk_cat", _hoc_ghk,
 "rates_cat", _hoc_rates,
 0, 0
};
#define KTF KTF_cat
#define alpmt alpmt_cat
#define alph alph_cat
#define betmt betmt_cat
#define beth beth_cat
#define efun efun_cat
#define ghk ghk_cat
 extern double KTF( _threadargsprotocomma_ double );
 extern double alpmt( _threadargsprotocomma_ double );
 extern double alph( _threadargsprotocomma_ double );
 extern double betmt( _threadargsprotocomma_ double );
 extern double beth( _threadargsprotocomma_ double );
 extern double efun( _threadargsprotocomma_ double );
 extern double ghk( _threadargsprotocomma_ double , double , double );
 /* declare global and static user variables */
#define USEGHK USEGHK_cat
 double USEGHK = 1;
#define a0m a0m_cat
 double a0m = 0.04;
#define a0h a0h_cat
 double a0h = 0.015;
#define erev erev_cat
 double erev = 100;
#define gmm gmm_cat
 double gmm = 0.1;
#define gmh gmh_cat
 double gmh = 0.6;
#define hmin hmin_cat
 double hmin = 10;
#define mmin mmin_cat
 double mmin = 0.2;
#define q10 q10_cat
 double q10 = 5;
#define vhalfm vhalfm_cat
 double vhalfm = -28;
#define vhalfh vhalfh_cat
 double vhalfh = -75;
#define zetam zetam_cat
 double zetam = 2;
#define zetah zetah_cat
 double zetah = 3.5;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gcatbar_cat", "mho/cm2",
 "ica_cat", "mA/cm2",
 "gcat_cat", "mho/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "q10_cat", &q10_cat,
 "mmin_cat", &mmin_cat,
 "hmin_cat", &hmin_cat,
 "a0h_cat", &a0h_cat,
 "zetah_cat", &zetah_cat,
 "vhalfh_cat", &vhalfh_cat,
 "gmh_cat", &gmh_cat,
 "a0m_cat", &a0m_cat,
 "zetam_cat", &zetam_cat,
 "vhalfm_cat", &vhalfm_cat,
 "gmm_cat", &gmm_cat,
 "USEGHK_cat", &USEGHK_cat,
 "erev_cat", &erev_cat,
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
"cat",
 "gcatbar_cat",
 0,
 "ica_cat",
 "gcat_cat",
 "hinf_cat",
 "htau_cat",
 "minf_cat",
 "mtau_cat",
 0,
 "m_cat",
 "h_cat",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 15, _prop);
 	/*initialize range parameters*/
 	gcatbar = 0.003;
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

 void _cat_mig_reg() {
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
 	ivoc_help("help ?1 cat /mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/cat_mig.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 static double FARADAY = 96520.0;
 static double R = 8.3134;
 static double KTOMV = .0853;
static int _reset;
static char *modelname = "T-calcium channel";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
   }
  return 0;
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
 _lalph = exp ( 0.0378 * zetah * ( _lv - vhalfh ) ) ;
   
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
 _lbeth = exp ( 0.0378 * zetah * gmh * ( _lv - vhalfh ) ) ;
   
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
 
static int  rates ( _threadargsprotocomma_ double _lv ) {
   double _la , _lb , _lqt ;
 _lqt = pow( q10 , ( ( celsius - 25.0 ) / 10.0 ) ) ;
   _la = 0.2 * ( - 1.0 * _lv + 19.26 ) / ( exp ( ( - 1.0 * _lv + 19.26 ) / 10.0 ) - 1.0 ) ;
   _lb = 0.009 * exp ( - _lv / 22.03 ) ;
   minf = _la / ( _la + _lb ) ;
   mtau = betmt ( _threadargscomma_ _lv ) / ( _lqt * a0m * ( 1.0 + alpmt ( _threadargscomma_ _lv ) ) ) ;
   if ( mtau < mmin ) {
     mtau = mmin ;
     }
   _la = 1.e-6 * exp ( - _lv / 16.26 ) ;
   _lb = 1.0 / ( exp ( ( - _lv + 29.79 ) / 10. ) + 1. ) ;
   hinf = _la / ( _la + _lb ) ;
   htau = beth ( _threadargscomma_ _lv ) / ( _lqt * a0h * ( 1.0 + alph ( _threadargscomma_ _lv ) ) ) ;
   if ( htau < hmin ) {
     htau = hmin ;
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
   gcat = gcatbar * m * m * h ;
   if ( USEGHK  == 1.0 ) {
     ica = gcat * ghk ( _threadargscomma_ v , cai , cao ) ;
     }
   else {
     ica = gcat * ( v - erev ) ;
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
static const char* nmodl_filename = "/mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/cat_mig.mod";
static const char* nmodl_file_text = 
  "TITLE T-calcium channel\n"
  ": T-type calcium channel\n"
  ": MODELDB 126814 CA3 by Safiulina et al - http://senselab.med.yale.edu/modeldb/ShowModel.asp?model=126814\n"
  ": by Michele Migliore\n"
  "\n"
  "\n"
  "UNITS {\n"
  "	(mA) = (milliamp)\n"
  "	(mV) = (millivolt)\n"
  "	(molar) = (1/liter)\n"
  "	(mM) = (millimolar)\n"
  "\n"
  "	FARADAY = 96520 (coul)\n"
  "	R = 8.3134 (joule/degC)\n"
  "	KTOMV = .0853 (mV/degC)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "	v (mV)\n"
  "	celsius = 25	(degC)\n"
  "	gcatbar=.003 (mho/cm2)\n"
  "	cai = 50.e-6 (mM)\n"
  "	cao = 2 (mM)\n"
  "	q10 = 5\n"
  "	mmin=0.2\n"
  "	hmin=10\n"
  "	a0h =0.015\n"
  "	zetah = 3.5\n"
  "	vhalfh = -75\n"
  "	gmh=0.6	\n"
  "	a0m =0.04\n"
  "	zetam = 2\n"
  "	vhalfm = -28\n"
  "	gmm=0.1	\n"
  "        USEGHK=1\n"
  "        erev = 100\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "	SUFFIX cat\n"
  "	USEION ca READ cai,cao WRITE ica\n"
  "        RANGE gcatbar, ica, gcat\n"
  "        RANGE hinf,minf,mtau,htau\n"
  "        GLOBAL USEGHK\n"
  "}\n"
  "\n"
  "STATE {\n"
  "	m h \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	ica (mA/cm2)\n"
  "        gcat (mho/cm2)\n"
  "	hinf\n"
  "	htau\n"
  "	minf\n"
  "	mtau\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	rates(v)\n"
  "	m = minf\n"
  "	h = hinf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "	SOLVE states METHOD cnexp\n"
  "	gcat = gcatbar*m*m*h\n"
  "        if (USEGHK == 1) {\n"
  "  	  ica = gcat*ghk(v,cai,cao)\n"
  "        } else {\n"
  "          ica = gcat*(v-erev)\n"
  "        }\n"
  "}\n"
  "\n"
  "DERIVATIVE states {	: exact when v held constant\n"
  "	rates(v)\n"
  "	m' = (minf - m)/mtau\n"
  "	h' = (hinf - h)/htau\n"
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
  "FUNCTION KTF(celsius (DegC)) (mV) {\n"
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
  "  alph = exp(0.0378*zetah*(v-vhalfh)) \n"
  "}\n"
  "\n"
  "FUNCTION beth(v(mV)) {\n"
  "  beth = exp(0.0378*zetah*gmh*(v-vhalfh)) \n"
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
  "PROCEDURE rates(v (mV)) { :callable from hoc\n"
  "	LOCAL a,b, qt\n"
  "        qt=q10^((celsius-25)/10)\n"
  "\n"
  "	a = 0.2*(-1.0*v+19.26)/(exp((-1.0*v+19.26)/10.0)-1.0)\n"
  "	b = 0.009*exp(-v/22.03)\n"
  "	minf = a/(a+b)\n"
  "	mtau = betmt(v)/(qt*a0m*(1+alpmt(v)))\n"
  "	if (mtau<mmin) {mtau=mmin}\n"
  "\n"
  "	a = 1.e-6*exp(-v/16.26)\n"
  "	b = 1/(exp((-v+29.79)/10.)+1.)\n"
  "	hinf = a/(a+b)\n"
  "	htau = beth(v)/(qt*a0h*(1+alph(v)))\n"
  "	if (htau<hmin) {htau=hmin}\n"
  "}\n"
  "\n"
  ;
#endif
