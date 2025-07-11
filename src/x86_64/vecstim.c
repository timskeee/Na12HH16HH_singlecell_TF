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
 
#define nrn_init _nrn_init__VecStim
#define _nrn_initial _nrn_initial__VecStim
#define nrn_cur _nrn_cur__VecStim
#define _nrn_current _nrn_current__VecStim
#define nrn_jacob _nrn_jacob__VecStim
#define nrn_state _nrn_state__VecStim
#define _net_receive _net_receive__VecStim 
#define element element__VecStim 
#define play play__VecStim 
 
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
#define index _p[0]
#define index_columnindex 0
#define etime _p[1]
#define etime_columnindex 1
#define space _p[2]
#define space_columnindex 2
#define v _p[3]
#define v_columnindex 3
#define _tsav _p[4]
#define _tsav_columnindex 4
#define _nd_area  *_ppvar[0]._pval
 
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
 static double _hoc_element(void*);
 static double _hoc_play(void*);
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
 "element", _hoc_element,
 "play", _hoc_play,
 0, 0
};
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 0,0
};
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
 static void _hoc_destroy_pnt(void* _vptr) {
   destroy_point_process(_vptr);
}
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"VecStim",
 0,
 0,
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
  if (nrn_point_prop_) {
	_prop->_alloc_seq = nrn_point_prop_->_alloc_seq;
	_p = nrn_point_prop_->param;
	_ppvar = nrn_point_prop_->dparam;
 }else{
 	_p = nrn_prop_data_alloc(_mechtype, 5, _prop);
 	/*initialize range parameters*/
  }
 	_prop->param = _p;
 	_prop->param_size = 5;
  if (!nrn_point_prop_) {
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
  }
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 
}
 static void _initlists();
 
#define _tqitem &(_ppvar[2]._pvoid)
 static void _net_receive(Point_process*, double*, double);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _vecstim_reg() {
	int _vectorized = 1;
  _initlists();
 	_pointtype = point_register_mech(_mechanism,
	 nrn_alloc,(void*)0, (void*)0, (void*)0, nrn_init,
	 hoc_nrnpointerindex, 1,
	 _hoc_create_pnt, _hoc_destroy_pnt, _member_func);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 5, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "area");
  hoc_register_dparam_semantics(_mechtype, 1, "pntproc");
  hoc_register_dparam_semantics(_mechtype, 2, "netsend");
 add_nrn_artcell(_mechtype, 2);
 add_nrn_has_net_event(_mechtype);
 pnt_receive[_mechtype] = _net_receive;
 pnt_receive_size[_mechtype] = 1;
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 VecStim /mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/vecstim.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
static int element(_threadargsproto_);
static int play(_threadargsproto_);
 
static void _net_receive (Point_process* _pnt, double* _args, double _lflag) 
{  double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _thread = (Datum*)0; _nt = (NrnThread*)_pnt->_vnt;   _p = _pnt->_prop->param; _ppvar = _pnt->_prop->dparam;
  if (_tsav > t){ extern char* hoc_object_name(); hoc_execerror(hoc_object_name(_pnt->ob), ":Event arrived out of order. Must call ParallelContext.set_maxstep AFTER assigning minimum NetCon.delay");}
 _tsav = t;   if (_lflag == 1. ) {*(_tqitem) = 0;}
 {
   if ( _lflag  == 1.0 ) {
     net_event ( _pnt, t ) ;
     }
   if ( _lflag  == 1.0  || _lflag  == 2.0 ) {
     element ( _threadargs_ ) ;
     if ( index > 0.0 ) {
       if ( etime - t >= 0.0 ) {
         artcell_net_send ( _tqitem, _args, _pnt, t +  etime - t , 1.0 ) ;
         }
       else {
         printf ( "Event in the stimulus vector at time %g is omitted since has value less than t=%g!\n" , etime , t ) ;
         artcell_net_send ( _tqitem, _args, _pnt, t +  0.0 , 2.0 ) ;
         }
       }
     }
   } }
 
/*VERBATIM*/
extern double* vector_vec();
extern int vector_capacity();
extern void* vector_arg();
 
static int  element ( _threadargsproto_ ) {
   
/*VERBATIM*/

  { void* vv; int i, size; double* px;
	i = (int)index;
	if (i >= 0) {
		vv = *((void**)(&space));
		if (vv) {
			size = vector_capacity(vv);
			px = vector_vec(vv);
			if (i < size) {
				etime = px[i];
				index += 1.;
			}else{
				index = -1.;
			}
		}else{
			index = -1.;
		}
	}
  }
  return 0; }
 
static double _hoc_element(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 element ( _p, _ppvar, _thread, _nt );
 return(_r);
}
 
static int  play ( _threadargsproto_ ) {
   
/*VERBATIM*/
	void** vv;
	vv = (void**)(&space);
	*vv = (void*)0;
	if (ifarg(1)) {
		*vv = vector_arg(1);
	}
  return 0; }
 
static double _hoc_play(void* _vptr) {
 double _r;
   double* _p; Datum* _ppvar; Datum* _thread; NrnThread* _nt;
   _p = ((Point_process*)_vptr)->_prop->param;
  _ppvar = ((Point_process*)_vptr)->_prop->dparam;
  _thread = _extcall_thread;
  _nt = (NrnThread*)((Point_process*)_vptr)->_vnt;
 _r = 1.;
 play ( _p, _ppvar, _thread, _nt );
 return(_r);
}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt) {
  int _i; double _save;{
 {
   index = 0.0 ;
   element ( _threadargs_ ) ;
   if ( index > 0.0 ) {
     if ( etime - t >= 0.0 ) {
       artcell_net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  etime - t , 1.0 ) ;
       }
     else {
       printf ( "Event in the stimulus vector at time %g is omitted since has value less than t=%g!\n" , etime , t ) ;
       artcell_net_send ( _tqitem, (double*)0, _ppvar[1]._pvoid, t +  0.0 , 2.0 ) ;
       }
     }
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
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, NrnThread* _nt, double _v){double _current=0.;v=_v;{
} return _current;
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
 v=_v;
{
}}

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/mnt/c/Users/Tim/Documents/Dev/BenShalom/netpyne/Na12HH16HH_singlecell_TF/mod/vecstim.mod";
static const char* nmodl_file_text = 
  ": $Id: vecstim.mod,v 1.3 2010/12/13 21:29:27 samn Exp $ \n"
  ":  Vector stream of events\n"
  "\n"
  "NEURON {\n"
  "  THREADSAFE\n"
  "       ARTIFICIAL_CELL VecStim \n"
  "}\n"
  "\n"
  "ASSIGNED {\n"
  "	index\n"
  "	etime (ms)\n"
  "	space\n"
  "}\n"
  "\n"
  "INITIAL {\n"
  "	index = 0\n"
  "	element()\n"
  "	if (index > 0) {\n"
  "		if (etime - t>=0) {\n"
  "			net_send(etime - t, 1)\n"
  "		} else {\n"
  "			printf(\"Event in the stimulus vector at time %g is omitted since has value less than t=%g!\\n\", etime, t)\n"
  "			net_send(0, 2)\n"
  "		}\n"
  "	}\n"
  "}\n"
  "\n"
  "NET_RECEIVE (w) {\n"
  "	if (flag == 1) { net_event(t) }\n"
  "	if (flag == 1 || flag == 2) {\n"
  "		element()\n"
  "		if (index > 0) {	\n"
  "			if (etime - t>=0) {\n"
  "				net_send(etime - t, 1)\n"
  "			} else {\n"
  "				printf(\"Event in the stimulus vector at time %g is omitted since has value less than t=%g!\\n\", etime, t)\n"
  "				net_send(0, 2)\n"
  "			}\n"
  "		}\n"
  "	}\n"
  "}\n"
  "\n"
  "VERBATIM\n"
  "extern double* vector_vec();\n"
  "extern int vector_capacity();\n"
  "extern void* vector_arg();\n"
  "ENDVERBATIM\n"
  "\n"
  "PROCEDURE element() {\n"
  "VERBATIM	\n"
  "  { void* vv; int i, size; double* px;\n"
  "	i = (int)index;\n"
  "	if (i >= 0) {\n"
  "		vv = *((void**)(&space));\n"
  "		if (vv) {\n"
  "			size = vector_capacity(vv);\n"
  "			px = vector_vec(vv);\n"
  "			if (i < size) {\n"
  "				etime = px[i];\n"
  "				index += 1.;\n"
  "			}else{\n"
  "				index = -1.;\n"
  "			}\n"
  "		}else{\n"
  "			index = -1.;\n"
  "		}\n"
  "	}\n"
  "  }\n"
  "ENDVERBATIM\n"
  "}\n"
  "\n"
  "PROCEDURE play() {\n"
  "VERBATIM\n"
  "	void** vv;\n"
  "	vv = (void**)(&space);\n"
  "	*vv = (void*)0;\n"
  "	if (ifarg(1)) {\n"
  "		*vv = vector_arg(1);\n"
  "	}\n"
  "ENDVERBATIM\n"
  "}\n"
  "        \n"
  "\n"
  ;
#endif
