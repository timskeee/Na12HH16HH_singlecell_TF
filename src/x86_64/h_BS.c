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
 
#define nrn_init _nrn_init__h
#define _nrn_initial _nrn_initial__h
#define nrn_cur _nrn_cur__h
#define _nrn_current _nrn_current__h
#define nrn_jacob _nrn_jacob__h
#define nrn_state _nrn_state__h
#define _net_receive _net_receive__h 
#define rate rate__h 
#define states states__h 
 
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
#define vhalfl _p[1]
#define vhalfl_columnindex 1
#define i _p[2]
#define i_columnindex 2
#define linf _p[3]
#define linf_columnindex 3
#define taul _p[4]
#define taul_columnindex 4
#define g _p[5]
#define g_columnindex 5
#define l _p[6]
#define l_columnindex 6
#define Dl _p[7]
#define Dl_columnindex 7
#define v _p[8]
#define v_columnindex 8
#define _g _p[9]
#define _g_columnindex 9
 
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
 static void _hoc_alpt(void);
 static void _hoc_bett(void);
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
 "setdata_h", _hoc_setdata,
 "alpt_h", _hoc_alpt,
 "bett_h", _hoc_bett,
 "rate_h", _hoc_rate,
 0, 0
};
#define alpt alpt_h
#define bett bett_h
 extern double alpt( _threadargsprotocomma_ double );
 extern double bett( _threadargsprotocomma_ double );
 /* declare global and static user variables */
#define a0t a0t_h
 double a0t = 0.009;
#define erev erev_h
 double erev = -37;
#define gmt gmt_h
 double gmt = 0.01;
#define kl kl_h
 double kl = -6;
#define qtl qtl_h
 double qtl = 1;
#define q10 q10_h
 double q10 = 4.5;
#define taumin taumin_h
 double taumin = 2;
#define vhalft vhalft_h
 double vhalft = -66.139;
#define zetat zetat_h
 double zetat = 20;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "erev_h", "mV",
 "vhalft_h", "mV",
 "a0t_h", "/ms",
 "zetat_h", "1",
 "gmt_h", "1",
 "taumin_h", "ms",
 "gbar_h", "mho/cm2",
 "vhalfl_h", "mV",
 "i_h", "mA/cm2",
 0,0
};
 static double delta_t = 0.01;
 static double l0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "erev_h", &erev_h,
 "kl_h", &kl_h,
 "vhalft_h", &vhalft_h,
 "a0t_h", &a0t_h,
 "zetat_h", &zetat_h,
 "gmt_h", &gmt_h,
 "q10_h", &q10_h,
 "qtl_h", &qtl_h,
 "taumin_h", &taumin_h,
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
 
#define _cvode_ieq _ppvar[0]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"h",
 "gbar_h",
 "vhalfl_h",
 0,
 "i_h",
 "linf_h",
 "taul_h",
 "g_h",
 0,
 "l_h",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 10, _prop);
 	/*initialize range parameters*/
 	gbar = 0.0001;
 	vhalfl = -78.474;
 	_prop->param = _p;
 	_prop->param_size = 10;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 1, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _h_BS_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 10, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 h /mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/h_BS.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "I-h channel from Magee 1998 for distal dendrites";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int rate(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[1], _dlist1[1];
 static int states(_threadargsproto_);
 
double alpt ( _threadargsprotocomma_ double _lv ) {
   double _lalpt;
 _lalpt = exp ( 0.0378 * zetat * ( _lv - vhalft ) ) ;
   
return _lalpt;
 }
 
static void _hoc_alpt(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  alpt ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
double bett ( _threadargsprotocomma_ double _lv ) {
   double _lbett;
 _lbett = exp ( 0.0378 * zetat * gmt * ( _lv - vhalft ) ) ;
   
return _lbett;
 }
 
static void _hoc_bett(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r =  bett ( _p, _ppvar, _thread, _nt, *getarg(1) );
 hoc_retpushx(_r);
}
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   rate ( _threadargscomma_ v ) ;
   Dl = ( linf - l ) / taul ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 rate ( _threadargscomma_ v ) ;
 Dl = Dl  / (1. - dt*( ( ( ( - 1.0 ) ) ) / taul )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   rate ( _threadargscomma_ v ) ;
    l = l + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / taul)))*(- ( ( ( linf ) ) / taul ) / ( ( ( ( - 1.0 ) ) ) / taul ) - l) ;
   }
  return 0;
}
 
static int  rate ( _threadargsprotocomma_ double _lv ) {
   double _la , _lqt ;
 _lqt = pow( q10 , ( ( celsius - 33.0 ) / 10.0 ) ) ;
   _la = alpt ( _threadargscomma_ _lv ) ;
   linf = 1.0 / ( 1.0 + exp ( - ( _lv - vhalfl ) / kl ) ) ;
   taul = bett ( _threadargscomma_ _lv ) / ( qtl * _lqt * a0t * ( 1.0 + _la ) ) + 1e-8 ;
   if ( taul < taumin ) {
     taul = taumin ;
     }
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
 
static int _ode_count(int _type){ return 1;}
 
static void _ode_spec(NrnThread* _nt, _Memb_list* _ml, int _type) {
   double* _p; Datum* _ppvar; Datum* _thread;
   Node* _nd; double _v; int _iml, _cntml;
  _cntml = _ml->_nodecount;
  _thread = _ml->_thread;
  for (_iml = 0; _iml < _cntml; ++_iml) {
    _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
    _nd = _ml->_nodelist[_iml];
    v = NODEV(_nd);
     _ode_spec1 (_p, _ppvar, _thread, _nt);
 }}
 
static void _ode_map(int _ieq, double** _pv, double** _pvdot, double* _pp, Datum* _ppd, double* _atol, int _type) { 
	double* _p; Datum* _ppvar;
 	int _i; _p = _pp; _ppvar = _ppd;
	_cvode_ieq = _ieq;
	for (_i=0; _i < 1; ++_i) {
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
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  l = l0;
 {
   rate ( _threadargscomma_ v ) ;
   l = linf ;
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   g = gbar * l ;
   i = g * ( v - erev ) ;
   }
 _current += i;

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
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
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
 {   states(_p, _ppvar, _thread, _nt);
  }}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = l_columnindex;  _dlist1[0] = Dl_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/h_BS.mod";
static const char* nmodl_file_text = 
  "TITLE I-h channel from Magee 1998 for distal dendrites\n"
  ": modified to take into account Sonia's exp. Apr.2008 M.Migliore\n"
  ": thread-safe 2010-05-18 Ben Suter\n"
  ": 2010-11-07 Ben Suter, removing \"hd\" from parameter names, changing suffix from \"hd\" to \"h\"\n"
  ": Parameters fit to pre-ZD current-clamp step responses from experiment BS0284 (traces and reconstruction from single corticospinal neuron)\n"
  ": 2011-09-18 Ben Suter, set default parameter values to those found from MRF optimization for BS0284 model\n"
  ":\n"
  ": :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
  ": Copyright 2011, Benjamin Suter (for changes only)\n"
  ": Used in model of corticospinal neuron BS0284 and published as:\n"
  ":  \"Intrinsic electrophysiology of mouse corticospinal neurons: a characteristic set of features embodied in a realistic computational model\"\n"
  ":  by Benjamin Suter, Michele Migliore, and Gordon Shepherd\n"
  ":  Submitted September 2011\n"
  ": :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
  "\n"
  "\n"
  "UNITS {\n"
  "    (mA) = (milliamp)\n"
  "    (mV) = (millivolt)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    v                       (mV)\n"
  "    celsius  = 34.0    (degC)\n"
  "    erev     = -37.0               (mV)\n"
  "    gbar     = 0.0001       (mho/cm2)\n"
  "    vhalfl   = -78.474      (mV)    : was -81\n"
  "    kl       = -6                   : was -8\n"
  "    vhalft   = -66.139      (mV)    : was -62\n"
  "    a0t      = 0.009        (/ms)   : was 0.0077696\n"
  "    zetat    = 20           (1)     : was 5\n"
  "    gmt      = 0.01         (1)     : was 0.057127\n"
  "    q10      = 4.5\n"
  "    qtl      = 1\n"
  "    taumin	= 2.0	(ms)		: minimal value of time constant\n"
  "}\n"
  "\n"
  "NEURON {\n"
  "    SUFFIX h\n"
  "    NONSPECIFIC_CURRENT i\n"
  "    RANGE gbar, vhalfl\n"
  "    RANGE linf, taul, g\n"
  "    GLOBAL taumin\n"
  "}\n"
  "\n"
  "STATE {\n"
  "    l\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    i       (mA/cm2)\n"
  "    linf\n"
  "    taul\n"
  "    g\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    rate(v)\n"
  "    l       = linf\n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE states METHOD cnexp\n"
  "    g       = gbar*l\n"
  "    i       = g*(v-erev)\n"
  "}\n"
  "\n"
  "FUNCTION alpt(v(mV)) {\n"
  "    alpt    = exp(0.0378*zetat*(v-vhalft))\n"
  "}\n"
  "\n"
  "FUNCTION bett(v(mV)) {\n"
  "    bett    = exp(0.0378*zetat*gmt*(v-vhalft))\n"
  "}\n"
  "\n"
  "DERIVATIVE states {     : exact when v held constant; integrates over dt step\n"
  "    rate(v)\n"
  "    l'      = (linf - l)/taul\n"
  "}\n"
  "\n"
  "PROCEDURE rate(v (mV)) { :callable from hoc\n"
  "    LOCAL a,qt\n"
  "    qt      = q10^((celsius-33)/10)\n"
  "    a       = alpt(v)\n"
  "    linf    = 1/(1 + exp(-(v-vhalfl)/kl))\n"
  "    taul    = bett(v)/(qtl*qt*a0t*(1+a)) + 1e-8 \n"
  "    if(taul < taumin) { taul = taumin } 	: min value of time constant\n"
  "}\n"
  ;
#endif
