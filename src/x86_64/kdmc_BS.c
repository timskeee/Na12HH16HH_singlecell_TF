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
 
#define nrn_init _nrn_init__kdmc
#define _nrn_initial _nrn_initial__kdmc
#define nrn_cur _nrn_cur__kdmc
#define _nrn_current _nrn_current__kdmc
#define nrn_jacob _nrn_jacob__kdmc
#define nrn_state _nrn_state__kdmc
#define _net_receive _net_receive__kdmc 
#define states states__kdmc 
#define trates trates__kdmc 
 
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
#define ik _p[1]
#define ik_columnindex 1
#define minf _p[2]
#define minf_columnindex 2
#define mtau _p[3]
#define mtau_columnindex 3
#define hinf _p[4]
#define hinf_columnindex 4
#define htau _p[5]
#define htau_columnindex 5
#define m _p[6]
#define m_columnindex 6
#define h _p[7]
#define h_columnindex 7
#define ek _p[8]
#define ek_columnindex 8
#define Dm _p[9]
#define Dm_columnindex 9
#define Dh _p[10]
#define Dh_columnindex 10
#define v _p[11]
#define v_columnindex 11
#define _g _p[12]
#define _g_columnindex 12
#define _ion_ek	*_ppvar[0]._pval
#define _ion_ik	*_ppvar[1]._pval
#define _ion_dikdv	*_ppvar[2]._pval
 
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
 static void _hoc_trates(void);
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
 "setdata_kdmc", _hoc_setdata,
 "trates_kdmc", _hoc_trates,
 0, 0
};
 /* declare global and static user variables */
#define a0h a0h_kdmc
 double a0h = 0.00058;
#define gmh gmh_kdmc
 double gmh = 0.2;
#define kh kh_kdmc
 double kh = 8;
#define km km_kdmc
 double km = 14;
#define q10 q10_kdmc
 double q10 = 3;
#define taumin taumin_kdmc
 double taumin = 0.1;
#define vhalfht vhalfht_kdmc
 double vhalfht = -100;
#define vhalfh vhalfh_kdmc
 double vhalfh = -5;
#define vhalfmt vhalfmt_kdmc
 double vhalfmt = -25;
#define zetah zetah_kdmc
 double zetah = 0.02;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "taumin_kdmc", "ms",
 "gbar_kdmc", "mho/cm2",
 "ik_kdmc", "mA/cm2",
 "mtau_kdmc", "ms",
 "htau_kdmc", "ms",
 0,0
};
 static double delta_t = 0.01;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "vhalfmt_kdmc", &vhalfmt_kdmc,
 "km_kdmc", &km_kdmc,
 "vhalfh_kdmc", &vhalfh_kdmc,
 "zetah_kdmc", &zetah_kdmc,
 "gmh_kdmc", &gmh_kdmc,
 "a0h_kdmc", &a0h_kdmc,
 "taumin_kdmc", &taumin_kdmc,
 "vhalfht_kdmc", &vhalfht_kdmc,
 "kh_kdmc", &kh_kdmc,
 "q10_kdmc", &q10_kdmc,
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
"kdmc",
 "gbar_kdmc",
 0,
 "ik_kdmc",
 "minf_kdmc",
 "mtau_kdmc",
 "hinf_kdmc",
 "htau_kdmc",
 0,
 "m_kdmc",
 "h_kdmc",
 0,
 0};
 static Symbol* _k_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 13, _prop);
 	/*initialize range parameters*/
 	gbar = 0.002;
 	_prop->param = _p;
 	_prop->param_size = 13;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 4, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_k_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ek */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ik */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dikdv */
 
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

 void _kdmc_BS_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("k", -10000.);
 	_k_sym = hoc_lookup("k_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 13, 4);
  hoc_register_dparam_semantics(_mechtype, 0, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "k_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 kdmc /mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/kdmc_BS.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "K-D channel with activation for motor cortex";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int trates(_threadargsprotocomma_ double);
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   trates ( _threadargscomma_ v ) ;
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 trates ( _threadargscomma_ v ) ;
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   trates ( _threadargscomma_ v ) ;
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
   }
  return 0;
}
 
static int  trates ( _threadargsprotocomma_ double _lv ) {
   double _lqt ;
 _lqt = pow( q10 , ( ( celsius - 34.0 ) / 10.0 ) ) ;
   minf = 1.0 / ( 1.0 + exp ( - ( _lv - vhalfmt ) / km ) ) ;
   mtau = 1.0 ;
   hinf = 1.0 / ( 1.0 + exp ( ( _lv - vhalfht ) / kh ) ) ;
   htau = exp ( zetah * gmh * ( _lv - vhalfh ) ) / ( _lqt * a0h * ( 1.0 + exp ( zetah * ( _lv - vhalfh ) ) ) ) ;
   if ( htau < taumin ) {
     htau = taumin ;
     }
    return 0; }
 
static void _hoc_trates(void) {
  double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   if (_extcall_prop) {_p = _extcall_prop->param; _ppvar = _extcall_prop->dparam;}else{ _p = (double*)0; _ppvar = (Datum*)0; }
  _thread = _extcall_thread;
  _nt = nrn_threads;
 _r = 1.;
 trates ( _p, _ppvar, _thread, _nt, *getarg(1) );
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
  ek = _ion_ek;
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
  ek = _ion_ek;
 _ode_matsol_instance1(_threadargs_);
 }}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_k_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_k_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_k_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
 {
   trates ( _threadargscomma_ v ) ;
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
  ek = _ion_ek;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ik = gbar * m * h * ( v - ek ) ;
   }
 _current += ik;

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
  ek = _ion_ek;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dik;
  _dik = ik;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dikdv += (_dik - ik)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ik += ik ;
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
  ek = _ion_ek;
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
static const char* nmodl_filename = "/mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/kdmc_BS.mod";
static const char* nmodl_file_text = 
  "TITLE K-D channel with activation for motor cortex\n"
  ": K-D current with activation, for motor cortex pyramidal neurons, per Miller et al. (2008)\n"
  ": Based on K-A current K-A current for Mitral Cells from Wang et al (1996), by M.Migliore Jan. 2002\n"
  ": 2011-02-25 Ben Suter, first version, using MM's kamt.mod as a starting template\n"
  ": 2011-09-18 Ben Suter, set default parameter values to those found from MRF optimization for BS0284 model\n"
  ":\n"
  ": :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
  ": Copyright 2011, Benjamin Suter\n"
  ": Used in model of corticospinal neuron BS0284 and published as:\n"
  ":  \"Intrinsic electrophysiology of mouse corticospinal neurons: a characteristic set of features embodied in a realistic computational model\"\n"
  ":  by Benjamin Suter, Michele Migliore, and Gordon Shepherd\n"
  ":  Submitted September 2011\n"
  ": :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\n"
  "\n"
  "\n"
  "NEURON {\n"
  "    THREADSAFE\n"
  "    SUFFIX kdmc\n"
  "    USEION k READ ek WRITE ik\n"
  "    RANGE  gbar, minf, mtau, hinf, htau, ik\n"
  "    GLOBAL taumin\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "    gbar    = 0.002     (mho/cm2)\n"
  "\n"
  "    celsius\n"
  "    ek                  (mV)   : must be explicitly def. in hoc\n"
  "    v                   (mV)\n"
  "\n"
  "    : activation\n"
  "    vhalfmt = -25       : original -20   : rough estimate from Miller et al (2008) Fig. 3D I-V curve\n"
  "    km      = 14        : manual fit to match this I-V curve\n"
  "\n"
  "    : inactivation\n"
  "    : NOTE: These values are still quite arbitrary (but get about the correct htau at -40 and -30 mV\n"
  "    vhalfh  = -5        : original -55\n"
  "    zetah   = 0.02      : original 0.05\n"
  "    gmh     = 0.2       : original 0.7\n"
  "    a0h     = 0.00058   : original 0.00055\n"
  "    taumin	= 0.1	(ms)		: minimal value of time constant\n"
  "\n"
  "    vhalfht = -100      : original -88   : measured by Storm (1988)\n"
  "    kh      = 8         : manual fit to match inactivation curve in Storm (1988) and Johnston+Wu textbook\n"
  "\n"
  "    q10     = 3\n"
  "}\n"
  "\n"
  "\n"
  "UNITS {\n"
  "    (mA) = (milliamp)\n"
  "    (mV) = (millivolt)\n"
  "    (pS) = (picosiemens)\n"
  "    (um) = (micron)\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "    ik      (mA/cm2)\n"
  "    minf        mtau (ms)\n"
  "    hinf        htau (ms)\n"
  "}\n"
  "\n"
  "\n"
  "STATE { m h }\n"
  "\n"
  "BREAKPOINT {\n"
  "    SOLVE states METHOD cnexp\n"
  "    ik  = gbar*m*h*(v - ek)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "    trates(v)\n"
  "    m   = minf\n"
  "    h   = hinf\n"
  "}\n"
  "\n"
  "DERIVATIVE states {\n"
  "    trates(v)\n"
  "    m'  = (minf-m)/mtau\n"
  "    h'  = (hinf-h)/htau\n"
  "}\n"
  "\n"
  "PROCEDURE trates(v) {\n"
  "    LOCAL qt\n"
  "    qt   = q10^((celsius-34)/10)\n"
  "\n"
  "    minf = 1/(1 + exp(-(v-vhalfmt)/km))\n"
  "    mtau = 1\n"
  "\n"
  "    hinf = 1/(1 + exp((v-vhalfht)/kh))\n"
  "    htau = exp(zetah*gmh*(v-vhalfh)) / (qt*a0h*(1 + exp(zetah*(v-vhalfh))))\n"
  "    if(htau < taumin) { htau = taumin } 	: min value of time constant\n"
  "}\n"
  ;
#endif
