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
 
#define nrn_init _nrn_init__nap
#define _nrn_initial _nrn_initial__nap
#define nrn_cur _nrn_cur__nap
#define _nrn_current _nrn_current__nap
#define nrn_jacob _nrn_jacob__nap
#define nrn_state _nrn_state__nap
#define _net_receive _net_receive__nap 
#define iassign iassign__nap 
#define rate rate__nap 
#define states states__nap 
 
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
#define gbar _p[0]
#define gbar_columnindex 0
#define DA_alphamshift _p[1]
#define DA_alphamshift_columnindex 1
#define DA_betamshift _p[2]
#define DA_betamshift_columnindex 2
#define DA_alphahfactor _p[3]
#define DA_alphahfactor_columnindex 3
#define DA_betahfactor _p[4]
#define DA_betahfactor_columnindex 4
#define m _p[5]
#define m_columnindex 5
#define h _p[6]
#define h_columnindex 6
#define ena _p[7]
#define ena_columnindex 7
#define Dm _p[8]
#define Dm_columnindex 8
#define Dh _p[9]
#define Dh_columnindex 9
#define ina _p[10]
#define ina_columnindex 10
#define minf _p[11]
#define minf_columnindex 11
#define hinf _p[12]
#define hinf_columnindex 12
#define mtau _p[13]
#define mtau_columnindex 13
#define htau _p[14]
#define htau_columnindex 14
#define g _p[15]
#define g_columnindex 15
#define v _p[16]
#define v_columnindex 16
#define _g _p[17]
#define _g_columnindex 17
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
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
 /* declaration of user functions */
 static void _hoc_hbet(void);
 static void _hoc_half(void);
 static void _hoc_iassign(void);
 static void _hoc_mbet(void);
 static void _hoc_malf(void);
 static void _hoc_rate(void);
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
 "setdata_nap", _hoc_setdata,
 "hbet_nap", _hoc_hbet,
 "half_nap", _hoc_half,
 "iassign_nap", _hoc_iassign,
 "mbet_nap", _hoc_mbet,
 "malf_nap", _hoc_malf,
 "rate_nap", _hoc_rate,
 0, 0
};
#define hbet hbet_nap
#define half half_nap
#define mbet mbet_nap
#define malf malf_nap
 extern double hbet( _threadargsprotocomma_ double );
 extern double half( _threadargsprotocomma_ double );
 extern double mbet( _threadargsprotocomma_ double );
 extern double malf( _threadargsprotocomma_ double );
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 "gbar_nap", 0, 1e+09,
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gbar_nap", "mho/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
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
 
#define _cvode_ieq _ppvar[3]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"nap",
 "gbar_nap",
 "DA_alphamshift_nap",
 "DA_betamshift_nap",
 "DA_alphahfactor_nap",
 "DA_betahfactor_nap",
 0,
 0,
 "m_nap",
 "h_nap",
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 18, _prop);
 	/*initialize range parameters*/
 	gbar = 0.0022;
 	DA_alphamshift = 0;
 	DA_betamshift = 0;
 	DA_alphahfactor = 0;
 	DA_betahfactor = 0;
 	_prop->param = _p;
 	_prop->param_size = 18;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
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

 void _nap_sidi_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 18, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 nap /mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/nap_sidi.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int iassign(_threadargsproto_);
static int rate(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
static int  iassign ( _threadargsproto_ ) {
   g = gbar * m * h ;
   ina = g * ( v - ena ) ;
    return 0; }
 
static void _hoc_iassign(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 iassign ( _p, _ppvar, _thread, _nt );
 hoc_retpushx(_r);
}
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rate ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rate ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rate ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
   }
  return 0;
}
 
double malf ( _threadargsprotocomma_ double _lv ) {
   double _lmalf;
 double _lva ;
 _lva = _lv + 12.0 + DA_alphamshift ;
   if ( fabs ( _lva ) < 1e-04 ) {
     _lva = _lva + 0.00001 ;
     }
   _lmalf = ( - 0.2816 * _lva ) / ( - 1.0 + exp ( - _lva / 9.3 ) ) ;
   
return _lmalf;
 }
 
static void _hoc_malf(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  malf ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double mbet ( _threadargsprotocomma_ double _lv ) {
   double _lmbet;
 double _lvb ;
 _lvb = _lv - 15.0 + DA_betamshift ;
   if ( fabs ( _lvb ) < 1e-04 ) {
     _lvb = _lvb + 0.00001 ;
     }
   _lmbet = ( 0.2464 * _lvb ) / ( - 1.0 + exp ( _lvb / 6.0 ) ) ;
   
return _lmbet;
 }
 
static void _hoc_mbet(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  mbet ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double half ( _threadargsprotocomma_ double _lv ) {
   double _lhalf;
 double _lvc ;
 _lvc = _lv + 42.8477 ;
   if ( fabs ( _lvc ) < 1e-04 ) {
     _lvc = _lvc + 0.00001 ;
     }
   _lhalf = ( 2.8e-5 + DA_alphahfactor ) * ( exp ( - _lvc / 4.0248 ) ) ;
   
return _lhalf;
 }
 
static void _hoc_half(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  half ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double hbet ( _threadargsprotocomma_ double _lv ) {
   double _lhbet;
 double _lvd ;
 _lvd = _lv - 413.9284 ;
   if ( fabs ( _lvd ) < 1e-04 ) {
     _lvd = _lvd + 0.00001 ;
     }
   _lhbet = ( 0.02 + DA_betahfactor ) / ( 1.0 + exp ( - _lvd / 148.2589 ) ) ;
   
return _lhbet;
 }
 
static void _hoc_hbet(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  hbet ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
static int  rate ( _threadargsprotocomma_ double _lv ) {
   double _lmsum , _lhsum , _lma , _lmb , _lha , _lhb ;
 _lma = malf ( _threadargscomma_ _lv ) ;
   _lmb = mbet ( _threadargscomma_ _lv ) ;
   _lha = half ( _threadargscomma_ _lv ) ;
   _lhb = hbet ( _threadargscomma_ _lv ) ;
   _lmsum = _lma + _lmb ;
   minf = _lma / _lmsum ;
   mtau = 1.0 / _lmsum ;
   _lhsum = _lha + _lhb ;
   hinf = _lha / _lhsum ;
   htau = 1.0 / _lhsum ;
    return 0; }
 
static void _hoc_rate(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 rate ( _p, _ppvar, _thread, _nt, *getarg(1) );
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
  ena = _ion_ena;
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
  ena = _ion_ena;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
 {
   rate ( _threadargscomma_ v ) ;
   m = minf ;
   h = hinf ;
   iassign ( _threadargs_ ) ;
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
  ena = _ion_ena;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   iassign ( _threadargs_ ) ;
   }
 _current += ina;

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
  ena = _ion_ena;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
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
  ena = _ion_ena;
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
static const char* nmodl_filename = "/mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/nap_sidi.mod";
static const char* nmodl_file_text = 
  ": Persistent Na+ channel\n"
  ": adapted from \n"
  ":  https://senselab.med.yale.edu/ModelDB/ShowModel.cshtml?model=144089&file=/PFCcell/mechanism/nap.mod\n"
  "\n"
  "NEURON {\n"
  "  SUFFIX nap\n"
  "  USEION na READ ena WRITE ina\n"
  "  RANGE gbar\n"
  "  RANGE DA_alphamshift,DA_betamshift\n"
  "  RANGE DA_alphahfactor, DA_betahfactor\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "  (mA) = (milliamp)\n"
  "  (mV) = (millivolt)	\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "  v (mV)\n"
  "  gbar= 0.0022 (mho/cm2) <0,1e9>\n"
  "  ena (mV) : do not set global ena here\n"
  "  DA_alphamshift=0 : 2 for 100% DA, 0 otherwise\n"
  "  DA_betamshift=0  : 5 for 100% DA,0 otherwise\n"
  "  DA_alphahfactor=0: -.8e-5 for DA, 0 otherwise\n"
  "  DA_betahfactor=0 : 0.014286-0.02 for DA, 0 otherwise\n"
  "}\n"
  "\n"
  "STATE {\n"
  "  m h\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "  ina (mA/cm2)\n"
  "  minf hinf \n"
  "  mtau (ms)\n"
  "  htau (ms)\n"
  "  g (mho/cm2)	\n"
  "}\n"
  "\n"
  "PROCEDURE iassign () {\n"
  "  g = gbar*m*h\n"
  "  ina = g*(v-ena)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "  rate(v)\n"
  "  m = minf\n"
  "  h = hinf\n"
  "  iassign()\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "  SOLVE states METHOD cnexp\n"
  "  iassign()\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "  rate(v)\n"
  "  m' = (minf-m)/mtau\n"
  "  h' = (hinf-h)/htau\n"
  "}\n"
  "\n"
  "UNITSOFF\n"
  "\n"
  "FUNCTION malf( v){ LOCAL va \n"
  "  va=v+12+DA_alphamshift\n"
  "  if (fabs(va)<1e-04){\n"
  "    va = va + 0.00001 \n"
  "  }\n"
  "  malf = (-0.2816*va)/(-1+exp(-va/9.3))	\n"
  "}\n"
  "\n"
  "FUNCTION mbet(v(mV))(/ms) { LOCAL vb \n"
  "  vb=v-15+DA_betamshift\n"
  "  if (fabs(vb)<1e-04){\n"
  "    vb = vb + 0.00001 \n"
  "  }  \n"
  "  mbet = (0.2464*vb)/(-1+exp(vb/6))\n"
  "}	\n"
  "\n"
  "FUNCTION half(v(mV))(/ms) { LOCAL vc \n"
  "  vc=v+42.8477\n"
  "  if (fabs(vc)<1e-04){\n"
  "    vc=vc+0.00001 \n"
  "  }\n"
  "  half= (2.8e-5+DA_alphahfactor)*(exp(-vc/4.0248))\n"
  "}\n"
  "\n"
  "FUNCTION hbet(v(mV))(/ms) { LOCAL vd\n"
  "  vd=v-413.9284\n"
  "  if (fabs(vd)<1e-04){\n"
  "    vd=vd+0.00001 \n"
  "  }\n"
  "  hbet= (0.02+DA_betahfactor)/(1+exp(-vd/148.2589))\n"
  "}\n"
  "\n"
  "PROCEDURE rate(v (mV)) {LOCAL msum, hsum, ma, mb, ha, hb\n"
  "  ma=malf(v) mb=mbet(v) ha=half(v) hb=hbet(v)\n"
  "	\n"
  "  msum = ma+mb\n"
  "  minf = ma/msum\n"
  "  mtau = 1/msum	\n"
  "	\n"
  "  hsum = ha+hb\n"
  "  hinf = ha/hsum\n"
  "  htau = 1/hsum\n"
  "}\n"
  "	\n"
  "UNITSON\n"
  ;
#endif
