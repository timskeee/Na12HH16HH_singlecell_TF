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
 
#define nrn_init _nrn_init__catt
#define _nrn_initial _nrn_initial__catt
#define nrn_cur _nrn_cur__catt
#define _nrn_current _nrn_current__catt
#define nrn_jacob _nrn_jacob__catt
#define nrn_state _nrn_state__catt
#define _net_receive _net_receive__catt 
#define states states__catt 
 
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
#define i _p[1]
#define i_columnindex 1
#define m _p[2]
#define m_columnindex 2
#define h _p[3]
#define h_columnindex 3
#define minf _p[4]
#define minf_columnindex 4
#define hinf _p[5]
#define hinf_columnindex 5
#define mtau _p[6]
#define mtau_columnindex 6
#define htau _p[7]
#define htau_columnindex 7
#define Dm _p[8]
#define Dm_columnindex 8
#define Dh _p[9]
#define Dh_columnindex 9
#define v _p[10]
#define v_columnindex 10
#define _g _p[11]
#define _g_columnindex 11
 
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
 "setdata_catt", _hoc_setdata,
 0, 0
};
 /* declare global and static user variables */
#define eca eca_catt
 double eca = 0;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "eca_catt", "mV",
 "gbar_catt", "mho/cm2",
 "i_catt", "mA/cm2",
 0,0
};
 static double delta_t = 1;
 static double h0 = 0;
 static double m0 = 0;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "eca_catt", &eca_catt,
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
"catt",
 "gbar_catt",
 0,
 "i_catt",
 0,
 "m_catt",
 "h_catt",
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 12, _prop);
 	/*initialize range parameters*/
 	gbar = 0;
 	_prop->param = _p;
 	_prop->param_size = 12;
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

 void _cat_traub_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 12, 1);
  hoc_register_dparam_semantics(_mechtype, 0, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 catt /mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/cat_traub.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "Calcium low threshold T type current for RD Traub, J Neurophysiol 89:909-921, 2003";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int states(_threadargsproto_);
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   minf = 1.0 / ( 1.0 + exp ( ( - v - 56.0 ) / 6.2 ) ) ;
   mtau = 0.204 + 0.333 / ( exp ( ( v + 15.8 ) / 18.2 ) + exp ( ( - v - 131.0 ) / 16.7 ) ) ;
   hinf = 1.0 / ( 1.0 + exp ( ( v + 80.0 ) / 4.0 ) ) ;
   if ( v < - 81.0 ) {
     htau = 0.333 * exp ( ( v + 466.0 ) / 66.6 ) ;
     }
   else {
     htau = 9.32 + 0.333 * exp ( ( - v - 21.0 ) / 10.5 ) ;
     }
   Dm = ( minf - m ) / mtau ;
   Dh = ( hinf - h ) / htau ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 minf = 1.0 / ( 1.0 + exp ( ( - v - 56.0 ) / 6.2 ) ) ;
 mtau = 0.204 + 0.333 / ( exp ( ( v + 15.8 ) / 18.2 ) + exp ( ( - v - 131.0 ) / 16.7 ) ) ;
 hinf = 1.0 / ( 1.0 + exp ( ( v + 80.0 ) / 4.0 ) ) ;
 if ( v < - 81.0 ) {
   htau = 0.333 * exp ( ( v + 466.0 ) / 66.6 ) ;
   }
 else {
   htau = 9.32 + 0.333 * exp ( ( - v - 21.0 ) / 10.5 ) ;
   }
 Dm = Dm  / (1. - dt*( ( ( ( - 1.0 ) ) ) / mtau )) ;
 Dh = Dh  / (1. - dt*( ( ( ( - 1.0 ) ) ) / htau )) ;
  return 0;
}
 /*END CVODE*/
 static int states (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
   minf = 1.0 / ( 1.0 + exp ( ( - v - 56.0 ) / 6.2 ) ) ;
   mtau = 0.204 + 0.333 / ( exp ( ( v + 15.8 ) / 18.2 ) + exp ( ( - v - 131.0 ) / 16.7 ) ) ;
   hinf = 1.0 / ( 1.0 + exp ( ( v + 80.0 ) / 4.0 ) ) ;
   if ( v < - 81.0 ) {
     htau = 0.333 * exp ( ( v + 466.0 ) / 66.6 ) ;
     }
   else {
     htau = 9.32 + 0.333 * exp ( ( - v - 21.0 ) / 10.5 ) ;
     }
    m = m + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / mtau)))*(- ( ( ( minf ) ) / mtau ) / ( ( ( ( - 1.0 ) ) ) / mtau ) - m) ;
    h = h + (1. - exp(dt*(( ( ( - 1.0 ) ) ) / htau)))*(- ( ( ( hinf ) ) / htau ) / ( ( ( ( - 1.0 ) ) ) / htau ) - h) ;
   }
  return 0;
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
 _ode_matsol_instance1(_threadargs_);
 }}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  h = h0;
  m = m0;
 {
   minf = 1.0 / ( 1.0 + exp ( ( - v - 56.0 ) / 6.2 ) ) ;
   mtau = 0.204 + 0.333 / ( exp ( ( v + 15.8 ) / 18.2 ) + exp ( ( - v - 131.0 ) / 16.7 ) ) ;
   hinf = 1.0 / ( 1.0 + exp ( ( v + 80.0 ) / 4.0 ) ) ;
   if ( v < - 81.0 ) {
     htau = 0.333 * exp ( ( v + 466.0 ) / 66.6 ) ;
     }
   else {
     htau = 9.32 + 0.333 * exp ( ( - v - 21.0 ) / 10.5 ) ;
     }
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   i = gbar * m * m * h * ( v - 125.0 ) ;
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
 _slist1[0] = m_columnindex;  _dlist1[0] = Dm_columnindex;
 _slist1[1] = h_columnindex;  _dlist1[1] = Dh_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/cat_traub.mod";
static const char* nmodl_file_text = 
  "TITLE Calcium low threshold T type current for RD Traub, J Neurophysiol 89:909-921, 2003\n"
  "\n"
  "COMMENT\n"
  "\n"
  "	Implemented by Maciej Lazarewicz 2003 (mlazarew@seas.upenn.edu)\n"
  "\n"
  "ENDCOMMENT\n"
  "\n"
  "INDEPENDENT { t FROM 0 TO 1 WITH 1 (ms) }\n"
  "\n"
  "UNITS { \n"
  "	(mV) = (millivolt) \n"
  "	(mA) = (milliamp) \n"
  "}\n"
  " \n"
  "NEURON { \n"
  "	SUFFIX catt\n"
  "	NONSPECIFIC_CURRENT i   : not causing [Ca2+] influx\n"
  "	RANGE gbar, i\n"
  "}\n"
  "\n"
  "PARAMETER { \n"
  "	gbar = 0.0 	(mho/cm2)\n"
  "	v eca 		(mV)  \n"
  "}\n"
  " \n"
  "ASSIGNED { \n"
  "	i 		(mA/cm2) \n"
  "	minf hinf 	(1)\n"
  "	mtau htau 	(ms) \n"
  "}\n"
  " \n"
  "STATE {\n"
  "	m h\n"
  "}\n"
  "\n"
  "BREAKPOINT { \n"
  "	SOLVE states METHOD cnexp\n"
  "	i = gbar * m * m * h * ( v - 125 ) \n"
  "}\n"
  " \n"
  "INITIAL { \n"
  "	minf  = 1 / ( 1 + exp( ( -v - 56 ) / 6.2 ) )\n"
  "	mtau  = 0.204 + 0.333 / ( exp( ( v + 15.8 ) / 18.2 ) + exp( ( - v - 131 ) / 16.7 ) )\n"
  "	hinf  = 1 / ( 1 + exp( ( v + 80 ) / 4 ) )\n"
  "	if( v < -81 ) {\n"
  "		htau  = 0.333 * exp( ( v + 466 ) / 66.6 )\n"
  "	}else{\n"
  "		htau  = 9.32 + 0.333 * exp( ( -v - 21 ) / 10.5 )\n"
  "	}\n"
  "	m  = minf\n"
  "	h  = hinf\n"
  "} \n"
  "\n"
  "DERIVATIVE states { \n"
  "	minf  = 1 / ( 1 + exp( ( -v - 56 ) / 6.2 ) )\n"
  "	mtau  = 0.204 + 0.333 / ( exp( ( v + 15.8 ) / 18.2 ) + exp( ( - v - 131 ) / 16.7 ) )\n"
  "	hinf  = 1 / ( 1 + exp( ( v + 80 ) / 4 ) )\n"
  "	if( v < -81 ) {\n"
  "		htau  = 0.333 * exp( ( v + 466 ) / 66.6 )\n"
  "	}else{\n"
  "		htau  = 9.32 + 0.333 * exp( ( -v - 21 ) / 10.5 )\n"
  "	}\n"
  "	m' = ( minf - m ) / mtau \n"
  "	h' = ( hinf - h ) / htau\n"
  "}\n"
  "\n"
  ;
#endif
