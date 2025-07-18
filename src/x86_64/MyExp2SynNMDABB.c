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
 
#define nrn_init _nrn_init__MyExp2SynNMDABB
#define _nrn_initial _nrn_initial__MyExp2SynNMDABB
#define nrn_cur _nrn_cur__MyExp2SynNMDABB
#define _nrn_current _nrn_current__MyExp2SynNMDABB
#define nrn_jacob _nrn_jacob__MyExp2SynNMDABB
#define nrn_state _nrn_state__MyExp2SynNMDABB
#define _net_receive _net_receive__MyExp2SynNMDABB 
#define state state__MyExp2SynNMDABB 
 
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
#define tau1NMDA _p[0]
#define tau1NMDA_columnindex 0
#define tau2NMDA _p[1]
#define tau2NMDA_columnindex 1
#define e _p[2]
#define e_columnindex 2
#define r _p[3]
#define r_columnindex 3
#define smax _p[4]
#define smax_columnindex 4
#define sNMDAmax _p[5]
#define sNMDAmax_columnindex 5
#define Vwt _p[6]
#define Vwt_columnindex 6
#define iNMDA _p[7]
#define iNMDA_columnindex 7
#define sNMDA _p[8]
#define sNMDA_columnindex 8
#define ica _p[9]
#define ica_columnindex 9
#define g _p[10]
#define g_columnindex 10
#define A2 _p[11]
#define A2_columnindex 11
#define B2 _p[12]
#define B2_columnindex 12
#define mgblock _p[13]
#define mgblock_columnindex 13
#define factor2 _p[14]
#define factor2_columnindex 14
#define cai _p[15]
#define cai_columnindex 15
#define cao _p[16]
#define cao_columnindex 16
#define DA2 _p[17]
#define DA2_columnindex 17
#define DB2 _p[18]
#define DB2_columnindex 18
#define v _p[19]
#define v_columnindex 19
#define _g _p[20]
#define _g_columnindex 20
#define _tsav _p[21]
#define _tsav_columnindex 21
#define _nd_area  *_ppvar[0]._pval
#define _ion_cai	*_ppvar[2]._pval
#define _ion_cao	*_ppvar[3]._pval
#define _ion_ica	*_ppvar[4]._pval
#define _ion_dicadv	*_ppvar[5]._pval
 
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
 static double _hoc_ghk(void*);
 static double _hoc_ghkg(void*);
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

 extern Prop* nrn_point_prop_;
 static int _pointtype;
 static void* _hoc_create_pnt(Object* _ho) { void* create_point_process(int, Object*);
 return create_point_process(_pointtype, _ho);
}
 static void _hoc_destroy_pnt(void*);
 static double _hoc_loc_pnt(void* _vptr) {double loc_point_process(int, void*);
 return loc_point_process(_pointtype, _vptr);
}
 static double _hoc_has_loc(void* _vptr) {double has_loc_point(void*);
 return has_loc_point(_vptr);
}
 static double _hoc_get_loc_pnt(void* _vptr) {
 double get_loc_point_process(void*); return (get_loc_point_process(_vptr));
}
 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata(void* _vptr) { Prop* _prop;
 _prop = ((Point_process*)_vptr)->_prop;
   _setdata(_prop);
 }
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 0,0
};
 static Member_func _member_func[] = {
 "loc", _hoc_loc_pnt,
 "has_loc", _hoc_has_loc,
 "get_loc", _hoc_get_loc_pnt,
 "ghk", _hoc_ghk,
 "ghkg", _hoc_ghkg,
 0, 0
};
#define ghk ghk_MyExp2SynNMDABB
#define ghkg ghkg_MyExp2SynNMDABB
 extern double ghk( _threadargsprotocomma_ double , double , double , double );
 extern double ghkg( _threadargsprotocomma_ double , double , double , double );
 /* declare global and static user variables */
#define fracca fracca_MyExp2SynNMDABB
 double fracca = 0.13;
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "tau1NMDA", "ms",
 "tau2NMDA", "ms",
 "e", "mV",
 "smax", "1",
 "sNMDAmax", "1",
 "A2", "1",
 "B2", "1",
 "iNMDA", "nA",
 "sNMDA", "1",
 "ica", "nA",
 "g", "umho",
 0,0
};
 static double A20 = 0;
 static double B20 = 0;
 static double delta_t = 0.01;
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 "fracca_MyExp2SynNMDABB", &fracca_MyExp2SynNMDABB,
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
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 
static int _ode_count(int);
static void _ode_map(int, double**, double**, double*, Datum*, double*, int);
static void _ode_spec(NrnThread*, _Memb_list*, int);
static void _ode_matsol(NrnThread*, _Memb_list*, int);
 
#define _cvode_ieq _ppvar[6]._i
 static void _ode_matsol_instance1(_threadargsproto_);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"MyExp2SynNMDABB",
 "tau1NMDA",
 "tau2NMDA",
 "e",
 "r",
 "smax",
 "sNMDAmax",
 "Vwt",
 0,
 "iNMDA",
 "sNMDA",
 "ica",
 "g",
 0,
 "A2",
 "B2",
 0,
 0};
 static Symbol* _ca_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 22, _prop);
 	/*initialize range parameters*/
 	tau1NMDA = 15;
 	tau2NMDA = 150;
 	e = 0;
 	r = 1;
 	smax = 1e+09;
 	sNMDAmax = 1e+09;
 	Vwt = 0;
  }
 	_prop->param = _p;
 	_prop->param_size = 22;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 7, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_ca_sym);
 nrn_promote(prop_ion, 1, 0);
 	_ppvar[2]._pval = &prop_ion->param[1]; /* cai */
 	_ppvar[3]._pval = &prop_ion->param[2]; /* cao */
 	_ppvar[4]._pval = &prop_ion->param[3]; /* ica */
 	_ppvar[5]._pval = &prop_ion->param[4]; /* _ion_dicadv */
 
}
 static void _initlists();
  /* some states have an absolute tolerance */
 static Symbol** _atollist;
 static HocStateTolerance _hoc_state_tol[] = {
 0,0
};
 static void _net_receive(Point_process*, double*, double);
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _MyExp2SynNMDABB_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("ca", -10000.);
 	_ca_sym = hoc_lookup("ca_ion");
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 22, 7);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 3, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 4, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 5, "ca_ion");
  hoc_register_dparam_semantics(_mechtype, 6, "cvodeieq");
 	hoc_register_cvode(_mechtype, _ode_count, _ode_map, _ode_spec, _ode_matsol);
 	hoc_register_tolerance(_mechtype, _hoc_state_tol, &_atollist);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 MyExp2SynNMDABB /mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/MyExp2SynNMDABB.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
 
#define FARADAY _nrnunit_FARADAY[_nrnunit_use_legacy_]
static double _nrnunit_FARADAY[2] = {0x1.78e555060882cp+16, 96485.3}; /* 96485.3321233100141 */
 
#define R _nrnunit_R[_nrnunit_use_legacy_]
static double _nrnunit_R[2] = {0x1.0a1013e8990bep+3, 8.3145}; /* 8.3144626181532395 */
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 
static int _ode_spec1(_threadargsproto_);
/*static int _ode_matsol1(_threadargsproto_);*/
 static int _slist1[2], _dlist1[2];
 static int state(_threadargsproto_);
 
double ghkg ( _threadargsprotocomma_ double _lv , double _lci , double _lco , double _lz ) {
   double _lghkg;
 double _lxi , _lf , _lexi , _lfxi ;
 _lf = R * ( celsius + 273.15 ) / ( _lz * ( 1e-3 ) * FARADAY ) ;
   _lxi = _lv / _lf ;
   _lexi = exp ( _lxi ) ;
   if ( fabs ( _lxi ) < 1e-4 ) {
     _lfxi = 1.0 - _lxi / 2.0 ;
     }
   else {
     _lfxi = _lxi / ( _lexi - 1.0 ) ;
     }
   _lghkg = _lf * ( ( _lci / _lco ) * _lexi - 1.0 ) * _lfxi ;
   
return _lghkg;
 }
 
static double _hoc_ghkg(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  ghkg ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 return(_r);
}
 
double ghk ( _threadargsprotocomma_ double _lv , double _lci , double _lco , double _lz ) {
   double _lghk;
 double _lxi , _lf , _lexi , _lfxi ;
 _lf = R * ( celsius + 273.15 ) / ( _lz * ( 1e-3 ) * FARADAY ) ;
   _lxi = _lv / _lf ;
   _lexi = exp ( _lxi ) ;
   if ( fabs ( _lxi ) < 1e-4 ) {
     _lfxi = 1.0 - _lxi / 2.0 ;
     }
   else {
     _lfxi = _lxi / ( _lexi - 1.0 ) ;
     }
   _lghk = ( .001 ) * _lz * FARADAY * ( _lci * _lexi - _lco ) * _lfxi ;
   
return _lghk;
 }
 
static double _hoc_ghk(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r =  ghk ( _p, _ppvar, _thread, _nt, *getarg(1) , *getarg(2) , *getarg(3) , *getarg(4) );
 return(_r);
}
 
/*CVODE*/
 static int _ode_spec1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {int _reset = 0; {
   DA2 = - A2 / tau1NMDA ;
   DB2 = - B2 / tau2NMDA ;
   }
 return _reset;
}
 static int _ode_matsol1 (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
 DA2 = DA2  / (1. - dt*( ( - 1.0 ) / tau1NMDA )) ;
 DB2 = DB2  / (1. - dt*( ( - 1.0 ) / tau2NMDA )) ;
  return 0;
}
 /*END CVODE*/
 static int state (double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) { {
    A2 = A2 + (1. - exp(dt*(( - 1.0 ) / tau1NMDA)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau1NMDA ) - A2) ;
    B2 = B2 + (1. - exp(dt*(( - 1.0 ) / tau2NMDA)))*(- ( 0.0 ) / ( ( - 1.0 ) / tau2NMDA ) - B2) ;
   }
  return 0;
}
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t; {
   double _lww ;
 _lww = _args[0] ;
   if ( r >= 0.0 ) {
       if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = A2;
    double __primary = (A2 + factor2 * _lww * r) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau1NMDA ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau1NMDA ) - __primary );
    A2 += __primary;
  } else {
 A2 = A2 + factor2 * _lww * r ;
       }
   if (nrn_netrec_state_adjust && !cvode_active_){
    /* discon state adjustment for cnexp case (rate uses no local variable) */
    double __state = B2;
    double __primary = (B2 + factor2 * _lww * r) - __state;
     __primary += ( 1. - exp( 0.5*dt*( ( - 1.0 ) / tau2NMDA ) ) )*( - ( 0.0 ) / ( ( - 1.0 ) / tau2NMDA ) - __primary );
    B2 += __primary;
  } else {
 B2 = B2 + factor2 * _lww * r ;
       }
 }
   } }
 
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
   nrn_update_ion_pointer(_ca_sym, _ppvar, 2, 1);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 3, 2);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 4, 3);
   nrn_update_ion_pointer(_ca_sym, _ppvar, 5, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
  A2 = A20;
  B2 = B20;
 {
   double _ltp ;
 Vwt = 0.0 ;
   if ( tau1NMDA / tau2NMDA > .9999 ) {
     tau1NMDA = .9999 * tau2NMDA ;
     }
   A2 = 0.0 ;
   B2 = 0.0 ;
   _ltp = ( tau1NMDA * tau2NMDA ) / ( tau2NMDA - tau1NMDA ) * log ( tau2NMDA / tau1NMDA ) ;
   factor2 = - exp ( - _ltp / tau1NMDA ) + exp ( - _ltp / tau2NMDA ) ;
   factor2 = 1.0 / factor2 ;
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
 _tsav = -1e20;
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
   double _liTOT ;
 mgblock = 1.0 / ( 1.0 + 0.28 * exp ( - 0.062 * v ) ) ;
   sNMDA = B2 - A2 ;
   if ( sNMDA > sNMDAmax ) {
     sNMDA = sNMDAmax ;
     }
   iNMDA = sNMDA * ( v - e ) * mgblock * ( 1.0 - fracca ) ;
   if ( fracca > 0.0 ) {
     ica = sNMDA * ghkg ( _threadargscomma_ v , cai , cao , 2.0 ) * mgblock * fracca ;
     }
   g = sNMDA * mgblock ;
   }
 _current += iNMDA;
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
  _ion_dicadv += (_dica - ica)/.001 * 1.e2/ (_nd_area);
 	}
 _g = (_g - _rhs)/.001;
  _ion_ica += ica * 1.e2/ (_nd_area);
 _g *=  1.e2/(_nd_area);
 _rhs *= 1.e2/(_nd_area);
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
 {   state(_p, _ppvar, _thread, _nt);
  } }}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
 _slist1[0] = A2_columnindex;  _dlist1[0] = DA2_columnindex;
 _slist1[1] = B2_columnindex;  _dlist1[1] = DB2_columnindex;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/MyExp2SynNMDABB.mod";
static const char* nmodl_file_text = 
  ": $Id: MyExp2SynNMDABB.mod,v 1.4 2010/12/13 21:28:02 samn Exp $ \n"
  "NEURON {\n"
  ":  THREADSAFE\n"
  "  POINT_PROCESS MyExp2SynNMDABB\n"
  "  RANGE e, i, iNMDA, s, sNMDA, r, tau1NMDA, tau2NMDA, Vwt, smax, sNMDAmax, g\n"
  "  NONSPECIFIC_CURRENT iNMDA\n"
  "  USEION ca READ cai,cao WRITE ica\n"
  "  GLOBAL fracca\n"
  "  RANGE ica\n"
  "}\n"
  "\n"
  "UNITS {\n"
  "  (nA) = (nanoamp)\n"
  "  (mV) = (millivolt)\n"
  "  (uS) = (microsiemens)\n"
  "  FARADAY = (faraday) (coulomb)\n"
  "  R = (k-mole) (joule/degC)\n"
  "}\n"
  "\n"
  "PARAMETER {\n"
  "  tau1NMDA = 15  (ms)\n"
  "  tau2NMDA = 150 (ms)\n"
  "  e        = 0	(mV)\n"
  "  r        = 1\n"
  "  smax     = 1e9 (1)\n"
  "  sNMDAmax = 1e9 (1)  \n"
  "  Vwt   = 0 : weight for inputs coming in from vector\n"
  "  fracca = 0.13 : fraction of current that is ca ions; Srupuston &al 95\n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "  v       (mV)\n"
  "  iNMDA   (nA)\n"
  "  sNMDA   (1)\n"
  "  mgblock (1)\n"
  "  factor2 (1)	\n"
  "  ica	  (nA)\n"
  "  cai     (mM)\n"
  "  cao     (mM)\n"
  "  g       (umho)\n"
  "}\n"
  "\n"
  "STATE {\n"
  "  A2 (1)\n"
  "  B2 (1)\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "  LOCAL tp\n"
  "  Vwt = 0 : testing\n"
  "  if (tau1NMDA/tau2NMDA > .9999) {\n"
  "    tau1NMDA = .9999*tau2NMDA\n"
  "  }\n"
  "  A2 = 0\n"
  "  B2 = 0	\n"
  "  tp = (tau1NMDA*tau2NMDA)/(tau2NMDA - tau1NMDA) * log(tau2NMDA/tau1NMDA)\n"
  "  factor2 = -exp(-tp/tau1NMDA) + exp(-tp/tau2NMDA)\n"
  "  factor2 = 1/factor2  \n"
  "}\n"
  "\n"
  "BREAKPOINT {\n"
  "  LOCAL iTOT\n"
  "  SOLVE state METHOD cnexp\n"
  "  : Jahr Stevens 1990 J. Neurosci\n"
  "  mgblock = 1.0 / (1.0 + 0.28 * exp(-0.062(/mV) * v) )\n"
  "  sNMDA = B2 - A2\n"
  "  if (sNMDA>sNMDAmax) {sNMDA=sNMDAmax}: saturation\n"
  "\n"
  "  :iTOT = sNMDA * (v - e) * mgblock  \n"
  "  :iNMDA = iTOT * (1-fracca)\n"
  "  :ica = iTOT * fracca\n"
  "  \n"
  "  iNMDA = sNMDA * (v - e) * mgblock * (1-fracca)\n"
  "  if(fracca>0.0){ica =   sNMDA * ghkg(v,cai,cao,2) * mgblock * fracca}\n"
  "  g = sNMDA * mgblock\n"
  "}\n"
  "\n"
  ":::INCLUDE \"ghk.inc\"\n"
  ":::realpath /mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/ghk.inc\n"
  "COMMENT\n"
  "    GHK function that returns effective driving force\n"
  "    Slope at low voltages is 1\n"
  "    z needs to be set as a PARAMETER\n"
  "ENDCOMMENT\n"
  "\n"
  "FUNCTION ghkg(v(mV), ci(mM), co(mM), z) (mV) {\n"
  "    LOCAL xi, f, exi, fxi\n"
  "    f = R*(celsius+273.15)/(z*(1e-3)*FARADAY)\n"
  "    xi = v/f\n"
  "    exi = exp(xi)\n"
  "    if (fabs(xi) < 1e-4) {\n"
  "        fxi = 1 - xi/2\n"
  "    }else{\n"
  "        fxi = xi/(exi - 1)\n"
  "    }\n"
  "    ghkg = f*((ci/co)*exi - 1)*fxi\n"
  "}\n"
  "\n"
  "FUNCTION ghk(v(mV), ci(mM), co(mM), z) (.001 coul/cm3) {\n"
  "    LOCAL xi, f, exi, fxi\n"
  "    f = R*(celsius+273.15)/(z*(1e-3)*FARADAY)\n"
  "    xi = v/f\n"
  "    exi = exp(xi)\n"
  "    if (fabs(xi) < 1e-4) {\n"
  "        fxi = 1 - xi/2\n"
  "    }else{\n"
  "        fxi = xi/(exi - 1)\n"
  "    }\n"
  "    ghk = (.001)*z*FARADAY*(ci*exi - co)*fxi\n"
  "}\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  "\n"
  ":::end INCLUDE ghk.inc\n"
  "\n"
  "DERIVATIVE state {\n"
  "  A2' = -A2/tau1NMDA\n"
  "  B2' = -B2/tau2NMDA\n"
  "}\n"
  "\n"
  "NET_RECEIVE(w (uS)) {LOCAL ww\n"
  "  ww=w\n"
  "  :printf(\"NMDA Spike: %g\\n\", t)\n"
  "  if(r>=0){ : if r>=0, g = NMDA*r\n"
  "    A2 = A2 + factor2*ww*r\n"
  "    B2 = B2 + factor2*ww*r\n"
  "  }\n"
  "}\n"
  ;
#endif
