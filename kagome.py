# This code is part of kagome-vqe.
#
# (C) Copyright LJSB Enterprises, LLC 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""IBM Open Challenge 2022 KAGOME VQE LATTICE tools


@author: rrodenbusch
"""
import sys
import pandas
import subprocess
import numpy as np
from sympy import init_printing
from time import time, sleep

init_printing(use_latex = True )

from qiskit_nature.problems.second_quantization.lattice import Lattice
from qiskit_ibm_runtime import Options
from qiskit_ibm_runtime import Session as RuntimeSession, Estimator as RuntimeEstimator

import rustworkx         as rx
version_info="myTools(major=1, minor=1, micro=0)"


def getVersion(output=True):
    myversion=f"Python: {str(sys.version_info)}\nkagome: {version_info}\n"
    if output:
        print(myversion)
    return(myversion)

""" ################################################################
        Lattrice Generation
    ################################################################ """
def create_lattice(numNodes,edges):
    # Generate graph of kagome unit cell
    graph = rx.PyGraph(multigraph=False)
    graph.add_nodes_from(range(numNodes))

    # Generate graph from the list of edges
    graph.add_edges_from(edges)

    # Make a Lattice from the graph
    return Lattice(graph)

def draw_lattice(lattice,positions=None,
                 with_labels=True,font_color='white',
                 node_color='purple', edge_color=None):
    style={'with_labels':with_labels, 'font_color':font_color,
           'node_color':node_color, }
    if edge_color is not None:
        style['edge_color'] = edge_color
    if positions is not None:
        style['pos']=positions
    lattice.draw(style=style)
    return style

def formatDuration(duration):
    days    = divmod(duration, 86400)        # Get days (without [0]!)
    hours   = divmod(days[1], 3600)          # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)           # Use remainder of hours to calc minutes
    msg = ''
    if int(days[0]) > 0:
        msg += f"{int(days[0])} days, {int(hours[0])} hours, {int(minutes[0])} min"
    elif int(hours[0]) > 0:
        msg += f"{int(hours[0])} hours, {int(minutes[0])} min"
    elif int(minutes[0]) > 0:
        msg += f"{int(minutes[0])} min, {int(minutes[1])} sec"
    else:
        msg += f"{np.around(minutes[1],5)} sec"
    return msg

def get_provider(hub='ibm-q', group='open', project='main', channel='ibm_quantum',
                 compatible=False, output=True, force=False):
    """ Get provider handle from account
            Args:
            Returns:
                Provider: Provider for current account
            Raises:
    """
    from qiskit import IBMQ
    if force or IBMQ.active_account() is None:
        IBMQ.load_account()

    if compatible:
        provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
        if output:
            print("Available backends")
            for curBackend in provider.backends():
                print(f"\t{curBackend.name()}")
        return provider
    else:
        from qiskit_ibm_provider import IBMProvider
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService(channel=channel)
        provider = IBMProvider(instance=f"{hub}/{group}/{project}")
        if output:
            print("Available backends")
            for curBackend in provider.backends():
                print(f"\t{curBackend.name}")
        return provider, service

def init_notebook(output=True):
    """Setup a jupyter notebook with personalized defaults and return Environment

        Args:
            output (Boolean): [True] Print the environment string
        Returns:
            str : Formatted environment string
    """
    import qiskit
    kVersion = f"qTools: {version_info}\nQiskit: {iter2str(dict(qiskit.__qiskit_version__))}"


    if output:
        print(kVersion)

    return(kVersion)

def iter2str(obj,name: str = None,indent: int = 4) -> str:
    """ iter2str
            Format an iterable into a printable string

        Args:
            obj (iterable): Object to be parsed
            name (str): Name of object to insert into string
            indent (str): The string to preprend to all new lines in the lookup

        Returns:
            str: The formatted string
        Raises:
        Examples:
    """
    if obj is None or not isinstance(obj,(tuple,set,list,dict)):
        raise TypeError(f"Argument must be dictionary, received {type(obj)}")
    if isinstance(indent,str):
        indent = len(indent)
    if not isinstance(indent,int):
        raise TypeError(f"Indent must be integer or string received {type(indent)}")

    if not name is None:
        iter_str = f"{name}: {type(obj).__name__}\n"
    else:
        iter_str = f"<{type(obj).__name__}>\n"

    pad = ' ' * indent
    if isinstance(obj,dict):
        for key, nxt_obj in obj.items():
            if isinstance(nxt_obj,(dict,set,tuple,list)):
                iter_str += iter2str(nxt_obj,name=f"{pad}{key}",indent=indent+4)
            else:
                iter_str += f"{pad}{key} : {nxt_obj}\n"
    elif isinstance(obj,(set,tuple,list)):
        idx = 0;
        for nxt_obj in obj:
            if isinstance(nxt_obj,(dict,set,tuple,list)):
                iter_str += iter2str(nxt_obj,name=f"{pad}{idx}",indent=indent+4)
            else:
                iter_str += f"{pad}{idx} -> {nxt_obj}\n"
            idx += 1
    else:
        iter_str += f"{pad}{obj}\n"

    return iter_str

""" ################################################################
        Class and utilities to run kagomeVQE data trials
    ################################################################ """
from qiskit.algorithms import MinimumEigensolver, VQEResult
from qiskit.providers import JobError
from qiskit_ibm_runtime.exceptions import (
    # QiskitRuntimeError,
    RuntimeJobFailureError,
    RuntimeInvalidStateError,
    IBMRuntimeError,
    RuntimeJobTimeoutError,
    RuntimeJobMaxTimeoutError,
)
exceptions_to_catch =(
    # QiskitRuntimeError,
    RuntimeJobFailureError,
    RuntimeInvalidStateError,
    IBMRuntimeError,
    RuntimeJobTimeoutError,
    RuntimeJobMaxTimeoutError,
    JobError,
    )
import numpy as np
import matplotlib.pyplot as plt
from qiskit.providers.jobstatus import JobStatus
_runningJobs = [JobStatus.INITIALIZING,
                JobStatus.QUEUED,
                JobStatus.VALIDATING,
                JobStatus.RUNNING, ]
_doneJobs    = [JobStatus.CANCELLED, JobStatus.DONE, JobStatus.ERROR, ]

# Define a custome VQE class to orchestra the ansatz, classical optimizers,
# initial point, callback, and final result
class KagomeResult(VQEResult):
    def __init__(self,data_dict=None):
        self._vqeResult = None
    @property
    def vqeResult(self):
        return self._vqeResult


class KagomeVQE(MinimumEigensolver):
    def __init__(self, estimator, circuit, optimizer, timeout=120, target=None,
                label=None, miniAnsatz = None, options=None, forced_xdata=None ):
        self._estimator       = estimator
        self._circuit         = circuit
        self._optimizer       = optimizer
        self._options         = options
        self._timeout         = timeout
        self._target          = target
        self._backend         = None
        self._session         = None
        # self._callback_data   = []
        # self._callback_points = []
        self._callback_fdata  = []
        self._callback_data   = []
        self._callback_xdata  = []
        self._callback_points = []
        self._job_results     = None
        self._resultList      = None
        self._result          = None
        self._initial_point   = None
        self._label           = label
        self._H               = None
        self._miniAnsatz      = None
        self._SPSA_callback_data = None
        self.attrs            = {}
        self._shots           = None
        self._forced_xdata    = forced_xdata
        self._xdata_idx       = 0
        if options is not None and options.get('shots',None) is not None:
            shots = options.get('shots',None)
        elif isinstance(options,Options):
            shots = options.runtime.shots

    def to_dict(self):
        myData = {}
        myData['_result']             = self._result
        myData['_label']              = self._label
        myData['_shots']              = self._shots
        myData['_initial_point']      = self._initial_point
        myData['_callback_data']      = self._callback_data
        myData['_callback_points']    = self._callback_points
        myData['_SPSA_callback_data'] = self._SPSA_callback_data
        myData['_target']             = self._target
        myData['attrs']               = self.attrs
        myData['job_results']         = self._job_results
        return myData

    def from_dict(self,myData):
        self._shots              = myData.get('_shots')
        self._result             = myData.get('_result',None)
        self._label              = myData.get('_label',None)
        self._initial_point      = myData.get('_initial_point',None)
        self._callback_data      = myData.get('_callback_data',None)
        self._callback_points    = myData.get('_callback_points',None)
        self._SPSA_callback_data = myData.get('_SPSA_callback_data')
        self._target             = myData.get('_target',None)
        self.attrs               = myData.get('attrs',None)
        self._job_results         = myData.get('job_results',None)

    def _callback(self, value):
        self._callback_data.append(value)

    def set_attr(self,key,value):
        self.attrs[key] = value

    def get_attr(self, key, default=None):
        return self.attrs.get(key,default)

    @property
    def SPSA_callback_data(self):
        return self._SPSA_callback_data

    @SPSA_callback_data.setter
    def SPSA_callback_data(self,value):
        self._SPSA_callback_data = value.copy()

    @property
    def callback_data(self):
        return self._callback_data

    @property
    def initial_point(self):
        return self._initial_point

    @property
    def label(self):
        return self._label

    @property
    def result(self):
        return self._result

    @property
    def shots(self):
        return self._shots

    @property
    def target(self) -> int | None:
        return self._target

    def __repr__(self):
        return(f"{self._label}"
               f"{self._target}\n"
               f"{self._initial_point}"
               f"{self._callback_data}"
               f"{self._result}")

    def __str__(self):
        return(f"{self._label}\n"
               f"Target: {self._target}\n"
               f"Result:\n{self._result}\n"
               f"InitialPoint:\n{self._initial_point}\n"
               f"Callback Data:\n{self._callback_data}\n")

    def list_result(self,prec=3):
        label = self.label.replace("\n",'')
        msg =  f"'{label}'\n"
        eigenvalue = self._result.eigenvalue if self._result is not None else None
        if eigenvalue is not None:
            msg += f"Computed: {np.around(self._result.eigenvalue,prec)}"
        results = parse_SPSA_callback(self.SPSA_callback_data)
        fdata = results.get('Fa', None)
        if fdata is not None and len(fdata) > 0:
            minVal = np.min(fdata)
            msg += f" Min {np.around(minVal,prec)}"
        if self._target is not None and eigenvalue is not None:
            min_error = abs((self._target - minVal) / self._target)
            rel_error = abs((self._target - self._result.eigenvalue) / self._target)
            msg += f" Target {np.around(self.target,prec)}"
            msg += f" Error {np.around(100*rel_error,prec)}%"
            msg += f" Min {np.around(100*min_error,prec)}%"
        return msg

    def show_result(self):
        eigenvalue = self._result.eigenvalue
        print(f'Computed ground state energy: {eigenvalue:.10f}')
        plt.title(self._label)
        plt.plot(self._callback_data, color='purple', lw=2)
        plt.ylabel('Energy')
        plt.xlabel('Iterations')
        if self._target is not None:
            rel_error = abs((self._target - eigenvalue) / self._target)
            print(f'Expected ground state energy: {self._target:.10f}')
            print(f'Relative error: {np.around(100*rel_error,8)}%')
            plt.axhline(y=self._target, color="tab:red", ls="--", lw=2, label="Target: " + str(self._target))
        else:
            plt.axhline(y=eigenvalue, color="tab:red", ls="--", lw=2, label="Target: None" )
        plt.legend()
        plt.grid()
        plt.show()

    def run(self, results=None):
        init_SPSA_callback()
        if self._service is None:
            if self._backend is None:
                self._result = self.compute_minimum_eigenvalue(self._H,x0=self._x0)
            else:
                self._result = self.compute_minimum_eigenvalue(self._H,x0=self._x0)
        else:
            with Session(service=self._service, backend=self._backend) as session:
                self._result = self.compute_minimum_eigenvalue(self._H,x0=self._x0)

        resultsList = self._resultsList if resultsList is None else resultsList
        if resultsList is not None:
            resultsList.append(self)
        return self._result

    def compute_minimum_eigenvalue(self, operators, aux_operators=None, x0=None, max_tries=5):
        # Define objective function to classically minimize over
        def objective(x):
            # Execute job with estimator primitive
            (try_count,job_result,start) = (0,None,time())
            while (job_result is None) and (try_count < max_tries):
                try_count+=1
                (jobID, job_status) = ('UNK', None )
                try:
                    job = self._estimator.run([self._circuit], [operators], [x])
                    jobId = job.job_id()
                    job_result = job.result()
                    job_status = job.status()
                    while (job_status in _runningJobs) and (try_count < max_tries):
                        duration = formatDuration(time()-start)
                        print(f"\nWAIT Job:{jobId} Status:{job_status} Try:{try_count} T:{duration}\n")
                        sleep(60)  # Avoid tight loop
                        job_result = job.result()
                        job_status = job.status()
                        if job_status is not JobStatus.DONE:
                            try_count+=1

                except exceptions_to_catch as ex:
                    job_status = 'Unknown' if job_status is None else job_status
                    duration = formatDuration(time()-start)
                    print(f"\nERROR cur Job:{jobId} Status:{job_status} Try:{try_count} T:{duration}\n")
                    if try_count < max_tries:
                        if job_status == 'Unknown':
                            print(f"\nSESSION Job {jobID} New RuntimeSession\n")
                            session = RuntimeSession(service=self._service, backend=self._backend)
                            self._estimator = RuntimeEstimator(session=session,
                                                               options=self._options)
                        pass
                    else:
                        print(f"\nERROR RETRIES Job {jobId} Try {try_count} Error {ex} Re-Raise\n")
                        raise

            value = None if job_result is None else job_result.values[0]
            # Save result information
            self._callback_data.append(value)
            self._callback_points.append(x)
            return value

        # Select an initial point for the ansatzs' parameters
        if x0 is None:
            x0 = np.pi/4 * np.random.rand(self._circuit.num_parameters)
        elif isinstance(x0,int) and (x0 == 0):
            x0 = np.zeros(self._circuit.num_parameters)

        self._initial_point = x0

        result = VQEResult()
        res = None
        # Run optimization
        start = time()
        try:
            res = self._optimizer.minimize(objective, x0=x0)
        except Exception as ex:
            print(f"Exception {type(ex)} ")
            # ex.printStackTrace()
            pass

        result.optimizer_time = time()-start
        if res is not None:
            result.cost_function_evals = res.nfev
            result.eigenvalue = res.fun
            result.optimal_parameters = res.x
            result.optimizer_result = res
        else:
            result.cost_function_evals = 0
            result.eigenvalue = 0
            result.optimal_parameters = None
            result.optimizer_result = None

        result.optimal_point = np.array(self._callback_data)
        self._result = result
        return result

    @property
    def result(self):
        return self._build_kagome_result(self._result)

    def _build_kagome_result(
            self,
            vqeResult :VQEResult,
            ) -> KagomeResult:
        result = KagomeResult(vqeResult=vqeResult)
        return result


""" ################################################################
             Manage the data generated for each test run
    ################################################################ """
def getX0(p_idx,curCache,last=False):
    expVal = None
    if isinstance(p_idx,int) and (p_idx >= 0) and (p_idx <= len(curCache)):
        spsa_data = parse_SPSA_callback(curCache[p_idx].SPSA_callback_data)
        Fdata = spsa_data['Fa']
        Xdata = spsa_data['Xa']
        if last:
            x0 = Xdata[-1]
            expVal = Fdata[-1]
        else:
            idx = np.argmin(Fdata)
            x0 = Xdata[idx]
            expVal = Fdata[idx]
    elif isinstance(p_idx,str) and (p_idx == '0'):
        x0 = 0
    else:
        x0 = None
    return x0, expVal

def list_results(data_cache, reverse=False):
    """ Display overview of result data cache
            Args:
                data_cache [list | KagomeVQE] : Data cache (or list of)
            Returns:
            Raises:
    """
    results = data_cache
    if not isinstance(data_cache,list):
        results = [data_cache]
    print_range = range(len(results))
    if reverse:
        print_range = reversed(print_range)
    for idx in print_range:
        kagomeVQE = results[idx]
        print(f"{idx}: {kagomeVQE.list_result()}\n")
    print("\n")

def load_results(fname):
    """ Save a copy of the results data cache
            Args:
                fname (str): filename with data data
            Returns:
            Raises:
    """
    import os.path
    results = []
    if os.path.isfile(fname):
        dict_results = load_object(fname)
        for curDict in dict_results:
            curVQE = KagomeVQE(None,None,None)
            curVQE.from_dict(curDict)
            results.append(curVQE)
    else:
        print(f"File not found.")
    print(f"Loaded {len(results)} results from {fname}")
    return results

def load_object(fname):
    import pickle, os.path
    if os.path.isfile(fname):
        with open(fname, 'rb') as obj_file:
            obj = pickle.load(obj_file)
            obj_file.close()
            return obj

def save_object(obj,fname):
    import pickle
    if obj is not None:
        with open(fname, 'wb') as obj_file:
            pickle.dump(obj, obj_file)
            obj_file.close()


def save_results(data_cache,fname):
    """ Save a copy of the results data cache
            Args:
                data_cache [list | KagomeVQE ]
                fname (str): filename to save data to
            Returns:
            Raises:
    """
    results = []
    if not isinstance(data_cache,list):
        data_cache = [data_cache]
    for curObj in data_cache:
        results.append(curObj.to_dict())
    # save the list
    save_object(results,fname)


""" ################################################################
   ---------------------- SPSA callback setup ----------------------
    Allocate and manage global data structure for SPSA direct callback
        _SPSA_callback_data : list
        SPSA_callback() : function for SPSA to callback
        get_SPSA_callback(): return a copy of the current callback data
        init_SPSA_callback(): clear the callback data cache
    ################################################################ """
_SPSA_callback_data = []
def SPSA_callback(nFuncs, x, Fx, stepSize, accepted):
    step_data = {}
    step_data['n'] = nFuncs
    step_data['x'] = x
    step_data['Fx'] = Fx
    step_data['stepSize'] = stepSize
    step_data['accepted'] = accepted
    _SPSA_callback_data.append(step_data)

def get_SPSA_callback():
    """ Return copy of SPSA callback data cache
            Args:
            Returns:
                list: Copy of current callback data
            Raises:
    """
    return _SPSA_callback_data.copy()

def init_SPSA_callback():
    """ Clear SPSA callback data cache
            Args:
            Returns:
            Raises:
    """
    _SPSA_callback_data.clear()

def plot_SPSA_convergence(curCache,indices=[-1],
                     conv_lim = 0.035,
                     movingAvg=5,perc=6,
                     scatter_xlim=(0,0.14),
                     convergence=False,
                     minStart=20):
    figsize=(8.5,6)
    fignum=1
    for idx in indices:
        #===== Get cached data =====#
        curResult = curCache[idx]
        target = curResult.target
        spsa_data = curResult.SPSA_callback_data
        parsed_data = parse_SPSA_convergence(spsa_data,
                                                    mavg=movingAvg,
                                                    target=target)
        # ==== Check for convergence and plot ====
        conv_idx=None
        if convergence:
            conv_idx = get_convergence_index(list(np.abs(parsed_data['avgSlopes'])),
                                             conv_lim=conv_lim,conv_ctr=5,offset=movingAvg)
            if conv_idx is not None:
                print(f"Convergence({conv_lim}) at {conv_idx} "
                      f"Fx={np.around(parsed_data['Fa'][conv_idx],perc)} "
                      f"{np.around(parsed_data['percErr'][conv_idx],perc)} % \n")
            else:
                print(f"Convergence Failure")
        plot_SPSA_callback(curResult,fignum=fignum,figsize=figsize,yline=conv_idx)

        if convergence:
            #==== Convergence multiplot ====#
            labels = ['slopes','avgSlopes', 'relF', 'percErr']
            yscale = 'log'
            plotData=[]
            for label in labels:
                data = parsed_data[label]
                if yscale == 'log':
                    data = list(np.abs(data))
                plotData.append(data)

            quick_plot(plotData, labels=labels,
                       title='Convergence', ylim=None,
                       fignum=fignum, figsize=figsize,
                       yscale=yscale)

            #==== Convergence Scatter Plots ====
            xdata = list(np.abs(parsed_data['avgSlopes']))[minStart:]
            xlabel='Slope Moving Average'
            ydata = list(np.abs(parsed_data['percErr']))[minStart:]
            ylabel='Percentage Error'
            quick_scatter(xdata,ydata,
                          xlabel=xlabel, ylabel=ylabel,
                          yscale='log',
                          xline=1.0, yline=conv_lim,
                          xlim=scatter_xlim,
                          fignum=fignum,figsize=figsize)

def quick_plot(data,labels=[],
               title=None, ylim=None, yscale='linear',
               xlabel='Iteration', ylabel='Value',
               fignum=1, figsize=(8.5,6),
               colors=['black','red','purple','green']
              ):

    if figsize is not None:
        plt.figure(fignum, figsize=figsize)
    data = data if isinstance(data[0],list) else [data]
    if title is not None:
        plt.title(title)
    if ylabel is not None:
        plt.ylabel(ylabel)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylim is not None:
        plt.ylim(ylim)
    plt.yscale(yscale)
    for idx in range(len(data)):
        label = labels[idx] if len(labels) > idx else ''
        curData = data[idx]
        color = colors[np.mod(idx,len(colors))]
        plt.plot(curData, color=color, lw=2, label=label)

    if labels is not None:
        plt.legend()
    plt.grid()
    return plt.show()

def quick_scatter(xdata,ydata,fignum=1,figsize=None,
                  xlim=None, ylim=None,
                  xline=None,yline=None,
                  yscale='linear',
                  xlabel='X Data', ylabel='Y Data'):
    if figsize is not None:
        plt.figure(fignum, figsize=figsize)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if xline is not None:
        plt.axhline(y=xline, color="tab:red", ls="--", lw=2, label="X {xline}" )
    if yline is not None:
        plt.axvline(x=yline, color="tab:red", ls="--", lw=2, label="Y {xline}" )
    plt.yscale(yscale)
    plt.scatter(xdata,ydata)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    return plt.show()

def get_convergence_index(xdata,conv_lim=0.035, conv_ctr=5, offset=1 ):
    conv_indices = np.where(np.array(xdata) < conv_lim )[0]
    #     print(f"found {len(conv_indices)} possibilities\n{conv_indices}")
    idx = offset-1
    conv_idx = None
    #     print(f"Starting search at idx={idx}")
    while (conv_idx is None) and (idx+conv_ctr < len(conv_indices)):
        targetList = list(range(conv_indices[idx],conv_indices[idx]+conv_ctr))
        testList = list(conv_indices[idx:idx+conv_ctr])
        if testList == targetList:
            conv_idx=conv_indices[idx+conv_ctr-1]
        idx+=1
    return conv_idx

def plot_SPSA_callback(obj,prec=6,fignum=1,figsize=None,yline=None):
    label = 'No Label'
    target = None
    gnd_state = 'Ground State: '
    duration = 'Unknown'
    nshots = '?'
    if isinstance(obj,list):
        spsa_data = obj
    elif isinstance(obj,dict):
        label = obj['label']
    else:
        label = obj._label
        target = obj._target
        nshots = obj._shots
        if hasattr(obj,'_result') and obj._result is not None:
            duration = formatDuration(obj._result.optimizer_time)
            eigenvalue = obj._result.eigenvalue
        else:
            duration = 10
            eigenvalue = None
        spsa_data = obj.SPSA_callback_data
        if spsa_data is None or len(spsa_data) == 0:
            print("No SPSA Callback Data Available")
            return

        gnd_state += f'Computed {np.around(eigenvalue,prec)} '

    if spsa_data is None or len(spsa_data) == 0:
        print(f"No spsa_data found")
        return

    parsed_data = parse_SPSA_callback(spsa_data)
    Xdata, Fdata, accepts = (parsed_data['Xa'], parsed_data['Fa'], parsed_data['accepts'])
    minEig = min(Fdata)
    if eigenvalue is None:
        eigenvalue = minEig
    gnd_state += f"Min {np.around(minEig,prec)} "
    print(f"Duration {duration} Shots={nshots} Iterations={accepts[0]+accepts[1]} Accepted="
          f"{np.around(100*accepts[0]/(accepts[0]+accepts[1]),1)} % "
          f"Rejected={accepts[1]} min at n={np.argmin(Fdata)}")

    if figsize is not None:
        plt.figure(fignum, figsize=figsize)
    plt.title(label)
    plt.plot(Fdata, color='purple', lw=2)
    plt.ylabel('Energy')
    plt.xlabel('Iterations')
    if target is not None:
        rel_error = abs((target - eigenvalue) / target)
        min_error = abs((target - minEig) / target)
        # gnd_state+= f'Expected {np.around(target,prec)}'
        print(f'Expected {np.around(target,prec)} {gnd_state}\n'
              f'Error {np.around(100*rel_error,prec)} % '
              f"Minimum {np.around(100*min_error,prec)} %")
        plt.axhline(y=target, color="tab:red", ls="--", lw=2,
                    label="Target: " + str(target))
    else:
        print(gnd_state)
        plt.axvhline(y=eigenvalue, color="tab:red", ls="--", lw=2, label="Target: None" )
    if yline is not None:
        plt.axvline(x=yline, color="tab:red", ls="--", lw=2, label=f"Y {yline}" )
    plt.legend()
    plt.grid()
    plt.show()

def parse_SPSA_callback(spsa_data):
    # Flatten the data for plotting
    (Fx,xvals,steps,accept,nF,Xdata,Fdata,Sdata,Ndata) = ([],[],[],[],[],[],[],[],[])
    Rdata = []
    accepts = [0,0]
    rejects = 0
    for curData in spsa_data:
        Rdata.append(rejects)
        Fx.append(curData['Fx'])
        xvals.append(curData['x'])
        steps.append(curData['stepSize'])
        accept.append(curData['accepted'])
        nF.append(curData['n'])
        if curData['accepted']:
            Xdata.append(curData['x'])
            Fdata.append(curData['Fx'])
            Sdata.append(curData['stepSize'])
            Ndata.append(curData['n'])
            accepts[0] += 1
            rejects = 0
        else:
            accepts[1] += 1
            rejects += 1
    return {'F':Fx, 'X':xvals, 'stepSize':steps,  'accepted':accept, 'nF':nF, 'accepts':accepts,
            'Fa':Fdata, 'Xa':Xdata, 'stepSizeA':Sdata, 'nFa':Ndata, 'Rdata': Rdata }

def _parse_SPSA_callback(spsa_data, data_list = ['Xa','Fa','accepts'] ):
    # Flatten the data for plotting
    data = _parse_SPSA_callback(spsa_data)
    return_data = []
    for key in data_list:
        return_data.append(data[key])
    return return_data


def parse_SPSA_convergence(spsa_data, mavg=5, target=None, details=False):
    parsed_data = parse_SPSA_callback(spsa_data)
    Fdata = parsed_data['Fa']
    steps = parsed_data['stepSizeA']
    (relF,delF,slopes,errs) = ([0],[0],[0], [0])  # Will be replaced with x[1] value for plotting
    for i in range(1,len(steps)):
        if target is not None:
            errs.append(100.0*(target - Fdata[i])/target)
        deltaF = Fdata[i] - Fdata[i-1]
        delF.append(deltaF)
        if steps[i] > 1e-10:
            slopes.append(deltaF/steps[i])
        elif details:
            print(f"Skipping iteration {i} with step={steps[i]}")
            slopes.append(slopes[-1])

        if Fdata[i] == 0:
            relF.append(relF[i-1])
        else:
            relF.append(deltaF/Fdata[i])

    if len(steps) > 1:
        relF[0] = relF[1]
        delF[0] = delF[1]
        slopes[0] = slopes[1]
        if target is not None:
            errs[0] = errs[1]
        else:
            errs = None

    # moving_average of absolute values of slopes for m elements
    mAvg = np.convolve( np.abs(slopes), np.ones(mavg), 'valid') / mavg
    mAvg = np.insert(mAvg,0,np.zeros(mavg-1)+0.001)

    parsed_data['delF'] = delF
    parsed_data['relF'] = relF
    parsed_data['slopes'] = slopes
    parsed_data['avgSlopes'] = mAvg
    parsed_data['percErr'] = errs
    return parsed_data


def try_estimator(estimator, A, H, x, retries=2):
    # Execute job with estimator primitive
    (job_result, try_count, start) = (None, 0, time())

    while (job_result is None) and (try_count < retries):
        try_count += 1
        jobId = 'UNK'
        try:
            job = estimator.run(A, H, x)
            jobId = job.job_id()
            job_status = job.status()
            if (job_status is not JobStatus.DONE) and (job_status not in _runningJobs):
                print(f"Job:{jobId} Try:{try_count} Status:{job_status} T:{time()-start}sec")
            else:
                job_result = job.result()

        except exceptions_to_catch as ex:
            print(f"cur Job:{jobId} Try:{try_count} Status:{job_status} T:{time()-start} sec")
            print(f"Job {jobId} Try {try_count} Error {ex}")
            if try_count < retries:
                pass
            else:
                print(f"Job {jobId} Try {try_count} Error {ex} Re-Raise")
                raise

    return job_result

def run_comparison(estimator, H, A, forced_xdata,
                   service=None, backend=None, label=None,
                   options=None ):
    from qiskit.primitives import Estimator
    from qiskit_ibm_runtime import Session, Options, Estimator as RuntimeEstimator

    (job_results,fdata,start_time) = ([],[],time())
    if label is None:
        label = f"Comparison"
    init_SPSA_callback()
    H = H if isinstance(H,list) else [H]
    A = A if isinstance(A,list) else [A]
    if service is None:
        if backend is None:
            estimator = Estimator(A, H, options=options )
        else:
            from qiskit.primitives import BackendEstimator
            estimator = BackendEstimator(backend, skip_transpilation=False,
                                         options=options)
        for curX in forced_xdata:
            job_result = try_estimator(estimator, A, H, [curX])
            if job_result is not None:
                fdata.append(job_result.values[0])
                job_results.append(job_result)
            else:
                raise JobError("Null Job Returned")
    else:
        if options is None:
            print(f"Default Options")
            options = Options()
        else:
            print(f"Provided Options")
        with Session(service=service, backend=backend) as session:
            label += f" {backend} "
            print(label)
            estimator = RuntimeEstimator(session=session, options=options)
            for curX in forced_xdata:
                job_result = try_estimator(estimator,A, H, [curX])
                if job_result is not None:
                    fdata.append(job_result.values[0])
                    job_results.append(job_result)
                else:
                    raise JobError("Null Job Returned")

    print(f"Runtime {formatDuration(time() - start_time)}")
    return fdata, job_results


def run_kagomeVQE(H, ansatz, optimizer, timeout=120, x0=None, target = None,
                 resultsList=None, service=None, backend=None, label=None,
                 miniAnsatz=None, options=None, forced_xdata=None):
    """ Run the eigenvalue search
            Args:
                H (SparsePauliSum): The Hamiltonian
                ansatz (QuantumCircuit):  Ansatz for the search
                optmizer (func): Chosen opmtimization routine
                timeout (int): Results wait timeout
                x0 (np.array): initial parameters
                target (float): Minimum eigenvalue (for labeling and err estimates)
                resultsList (list): TBD
                service (Service): Service to use for runtime execution
                backend (Union[str, Backend]): Backend to use for runtime exections
                label (str): Initial label
                miniAnsatz (QuantumCircuit): ansatz without any ancillary qubits
                shots (int): Run option for shots
            Returns:
                KagomeVQE: Custom class with results
            Raises:
    """
    if label is None:
        label = f"KagomeVQE {optimizer.__class__.__name__}"
    if resultsList is not None:
        label += f" idx={len(resultsList)}"

    init_SPSA_callback()

    if service is None:
        if backend is None:
            from qiskit.primitives import Estimator

            estimator = Estimator([ansatz], [H], options=options )
            print(label)
            kagomeVQE = KagomeVQE(estimator, ansatz, optimizer,
                                   timeout=None, target=target, label=label,
                                   miniAnsatz=miniAnsatz, forced_xdata=forced_xdata )
            if forced_xdata is None:
                result = kagomeVQE.compute_minimum_eigenvalue(H,x0=x0)
            else:
                print(f"Running a forced data set on Local")
                result, job_results = run_comparison(estimator, H, ansatz, forced_xdata,
                                        service=None, backend=None, label=label,
                                        options=options,)
                kagomeVQE._callback_points = forced_xdata
                kagomeVQE._callback_data = result
                kagomeVQE._job_results = job_results
        else:
            from qiskit.primitives import BackendEstimator
            estimator = BackendEstimator(backend, skip_transpilation=False,
                                         options=options)
            print(label)
            kagomeVQE = KagomeVQE(estimator, ansatz, optimizer,
                                  timeout=None, target=target, label=label,
                                  miniAnsatz=miniAnsatz, forced_xdata=forced_xdata )
            if forced_xdata is None:
                result = kagomeVQE.compute_minimum_eigenvalue(H,x0=x0)
            else:
                print(f"Running a forced data set on Local Simulator")
                result, job_results = run_comparison(estimator, H, ansatz, forced_xdata,
                                             service=service, backend=backend, label=label,
                                             options=options,)
                kagomeVQE._callback_points = forced_xdata
                kagomeVQE._callback_data = result
                kagomeVQE._job_results = job_results
    else:
        from qiskit_ibm_runtime import Session, Options, Estimator as RuntimeEstimator
        if options is None:
            print(f"Default Options")
            options = Options()
        else:
            print(f"Provided Options")
        with Session(service=service, backend=backend) as session:
            label += f" {backend} {optimizer.__class__.__name__}"
            print(label)
            estimator = RuntimeEstimator(session=session, options=options)
            # , options=options)
            kagomeVQE = KagomeVQE(estimator, ansatz, optimizer,
                                  timeout=None, target=target, label=label,
                                  miniAnsatz=miniAnsatz, forced_xdata=forced_xdata)
            if forced_xdata is None:
                kagomeVQE._backend = backend
                kagomeVQE._service = service
                result = kagomeVQE.compute_minimum_eigenvalue(H,x0=x0)
            else:
                print(f"Running a forced data set on RunTime Simulator")
                result, job_results = run_comparison(estimator, H, ansatz, forced_xdata,
                                             service=service, backend=backend, label=label,
                                             options=options,)
                kagomeVQE._callback_points = forced_xdata
                kagomeVQE._callback_data = result
                kagomeVQE._job_results = job_results

    if resultsList is not None:
        resultsList.append(kagomeVQE)

    kagomeVQE.SPSA_callback_data = get_SPSA_callback().copy()
    kagomeVQE.H = H

    if hasattr(result,'optimizer_time'):
        print(f"Runtime {formatDuration(result.optimizer_time)}")
    return kagomeVQE


def setup_kagomeJob(H, ansatz, optimizer, timeout=120, x0=None, target = None,
                 resultsList=None, service=None, backend=None, label=None,
                 miniAnsatz=None, shots=1024, options=None):
    """ Run the eigenvalue search
            Args:
                H (SparsePauliSum): The Hamiltonian
                ansatz (QuantumCircuit):  Ansatz for the search
                optmizer (func): Chosen opmtimization routine
                timeout (int): Results wait timeout
                x0 (np.array): initial parameters
                target (float): Minimum eigenvalue (for labeling and err estimates)
                resultsList (list): TBD
                service (Service): Service to use for runtime execution
                backend (Union[str, Backend]): Backend to use for runtime exections
                label (str): Initial label
                miniAnsatz (QuantumCircuit): ansatz without any ancillary qubits
                shots (int): Run option for shots
                options (dict): Runtime options
            Returns:
                KagomeVQE: Custom class with results
            Raises:
    """
    if label is None:
        label = f"KagomeVQE {optimizer.__class__.__name__}"
    if resultsList is not None:
        label += f" idx={len(resultsList)}"

    if service is None:
        if backend is None:
            from qiskit.primitives import Estimator
            estimator = Estimator([ansatz], [H], options=options )
            kagomeVQE = KagomeVQE(estimator, ansatz, optimizer,
                                   timeout=None, target=target, label=label,
                                   shots = shots )
                                         # options={'shots':shots})
        else:
            from qiskit.primitives import BackendEstimator
            estimator = BackendEstimator(backend, skip_transpilation=False,
                                         options=options)
                                         # options={'shots':shots})
            kagomeVQE = KagomeVQE(estimator, ansatz, optimizer,
                                  timeout=None, target=target, label=label,
                                  options=options, miniAnsatz=miniAnsatz )
    else:
        from qiskit_ibm_runtime import Session, Estimator as RuntimeEstimator
        estimator = RuntimeEstimator(session=session,
                                     options=options )
                                     # options={'shots':shots})
        kagomeVQE = KagomeVQE(estimator, ansatz, optimizer,
                              timeout=None, target=target, label=label,
                              shots=shots)
        label += f" {backend} {optimizer.__class__.__name__}"



    kagomeVQE.set_attr('SPSA_callback_data',get_SPSA_callback())
    kagomeVQE.set_attr('H',H)

    print(f"Runtime {formatDuration(result.optimizer_time)}")
    return kagomeVQE

def mytime():
    """ epoch
            Return integer seconds of Unix epoch time
        Args:
        Returns:
            int: Seconds since midnight 1/1/1970
        Raises:
    """
    from time import time
    return(int(time()))

def strtime(epoch=None):
    """ epochString
            Return formatted time string in UTC
        Args:
            epoch: int time, if None current time is used
        Returns:
            str: Formatted time string
        Raises:
    """
    from time import gmtime,asctime
    if epoch is None:
        epoch = mytime()
    return(f"{asctime(gmtime(epoch))} UTC")

if __name__ == "__main__":
    verstr = getVersion(output=True)

def setup_devices(max_parallel=3):
    from qiskit_ibm_runtime import (QiskitRuntimeService, Session, Options,
                                    Estimator as RuntimeEstimator)
    from qiskit_aer import AerSimulator, StatevectorSimulator

    from qiskit.providers.fake_provider import FakeGuadalupe

    svectorSim   = StatevectorSimulator(max_parallel_experiments=max_parallel)

    providerV0 = get_provider(channel='ibm_quantum',
                                            hub='ibm-q', group='open', project='main',
                                            compatible=True,
                                            output=True,   # Print out the available backends
                                           )


    # 16 qubit
    guadalupeSim = AerSimulator(max_parallel_experiments=max_parallel).from_backend(FakeGuadalupe())

    ###### ibm Version Compatible backends ##########################
    # 5 qubit H/W
    belemV0            = providerV0.get_backend('ibmq_belem')
    limaV0             = providerV0.get_backend('ibmq_lima')
    manilaV0           = providerV0.get_backend('ibmq_manila')
    quitoV0            = providerV0.get_backend('ibmq_quito')
    # 7 qubit H/W
    nairobiV0          = providerV0.get_backend('ibm_nairobi')
    osloV0             = providerV0.get_backend('ibm_oslo')
    jakartaV0          = providerV0.get_backend('ibmq_jakarta')
    perthV0            = providerV0.get_backend('ibm_perth')
    lagosV0            = providerV0.get_backend('ibm_lagos')
    backendsV0 = {'5':belemV0,  '7':osloV0, '16':guadalupeSim,
                  'm':manilaV0, 'q':quitoV0, 'a':limaV0, 'b':belemV0,
                  'j':jakartaV0,'o':osloV0,  'l':lagosV0,
                  'n':nairobiV0,'p':perthV0,
                  'g':FakeGuadalupe(),}


    #################### Runtime API Backends ###################################
    provider,service = get_provider(channel='ibm_quantum',
                                            hub='ibm-q', group='open', project='main',
                                            compatible=False,
                                            output=True,   # Print out the available backends
                                           )
    backends_runtime = {'5':'ibmq_belem',  '7':'oslo',
                        'm':'ibmq_manila', 'q':'ibmq_quito', 'a':'ibmq_lima', 'b':'ibmq_belem',
                        'j':'ibmq_jakarta','o':'ibm_oslo',  'l':'ibm_lagos',
                        'n':'ibm_nairobi','p':'ibm_perth',
                        'g':'ibmq_guadalupe',}
    # 5 qubit H/W
    belem            = provider.get_backend('ibmq_belem')
    lima             = provider.get_backend('ibmq_lima')
    manila           = provider.get_backend('ibmq_manila')
    quito            = provider.get_backend('ibmq_quito')
    # 7 qubit H/W
    nairobi          = provider.get_backend('ibm_nairobi')
    oslo             = provider.get_backend('ibm_oslo')
    jakarta          = provider.get_backend('ibmq_jakarta')
    perth            = provider.get_backend('ibm_perth')
    lagos            = provider.get_backend('ibm_lagos')

    backends = {'5':belem,  '7':oslo, '16':guadalupeSim,
                'm':manila, 'q':quito, 'a':lima, 'b':belem,
                'j':jakarta,'o':oslo,  'l':lagos,
                'n':nairobi,'p':perth,
                'g':guadalupeSim,}


    # 5qubit
    belemSim     = AerSimulator(max_parallel_experiments=max_parallel).from_backend(belem)
    limaSim      = AerSimulator(max_parallel_experiments=max_parallel).from_backend(lima)
    quitoSim     = AerSimulator(max_parallel_experiments=max_parallel).from_backend(quito)
    manilaSim    = AerSimulator(max_parallel_experiments=max_parallel).from_backend(manila)

    # 7 qubit
    nairobiSim   = AerSimulator(max_parallel_experiments=max_parallel).from_backend(nairobi)
    osloSim      = AerSimulator(max_parallel_experiments=max_parallel).from_backend(oslo)
    jakartaSim   = AerSimulator(max_parallel_experiments=max_parallel).from_backend(jakarta)
    perthSim     = AerSimulator(max_parallel_experiments=max_parallel).from_backend(perth)
    lagosSim     = AerSimulator(max_parallel_experiments=max_parallel).from_backend(lagos)

    simulators = {'m':manilaSim,    'q':quitoSim, 'a':limaSim, 'b':belemSim,
                  'j':jakartaSim,   'o':osloSim,  'l':lagosSim,
                  'n':nairobiSim,   'p':perthSim,
                  'g':guadalupeSim,
                 }
    print(f"Hardware Setup Complete")
    return provider, service, backends, simulators, backendsV0, backends_runtime

def init_optimizers():
    optimizers={}
    from qiskit.algorithms.optimizers import SPSA
    optimizers['00'] = {'opt': SPSA(maxiter=5), 'label':'SPSA(5)' }
    optimizers['01'] = {'opt': SPSA(maxiter=75), 'label':'SPSA(75)' }
    optimizers['02'] = {'opt': SPSA(maxiter=75,callback=SPSA_callback),
                        'label':'SPSA(75,cb)' }
    optimizers['03'] = {'opt': SPSA(maxiter=75,callback=SPSA_callback,blocking=True),
                        'label':'SPSA(75,cb,block)' }
    optimizers['04'] = {'opt': SPSA(maxiter=150,callback=SPSA_callback),
                        'label':'SPSA(150,cb)' }
    optimizers['05'] = {'opt': SPSA(maxiter=150,callback=SPSA_callback,blocking=True),
                        'label':'SPSA(150,cb,block)' }
    optimizers['06'] = {'opt': SPSA(maxiter=300,callback=SPSA_callback),
                        'label':'SPSA(300,cb)' }
    optimizers['07'] = {'opt': SPSA(maxiter=300,callback=SPSA_callback,blocking=True),
                        'label':'SPSA(300,cb,block)' }
    optimizers['08'] = {'opt': SPSA(maxiter=150,callback=SPSA_callback,trust_region=True),
                        'label':'SPSA(150,cb,trust)' }
    optimizers['09'] = {'opt': SPSA(maxiter=300,callback=SPSA_callback,trust_region=True),
                        'label':'SPSA(300,cb,trust)' }
    optimizers['10'] = {'opt': SPSA(maxiter=300,callback=SPSA_callback,second_order=True),
                        'label':'SPSA(300,cb,O(2)' }
    optimizers['11'] = {'opt': SPSA(maxiter=100,callback=SPSA_callback),
                        'label':'SPSA(100,cb)' }
    optimizers['12'] = {'opt': SPSA(maxiter=100,callback=SPSA_callback,trust_region=True),
                        'label':'SPSA(100,cb,trust)' }
    optimizers['13'] = {'opt': SPSA(maxiter=350,callback=SPSA_callback),
                        'label':'SPSA(350,cb)' }
    optimizers['14'] = {'opt': SPSA(maxiter=200,callback=SPSA_callback),
                        'label':'SPSA(200,cb)' }
    optimizers['15'] = {'opt': SPSA(maxiter=100,callback=SPSA_callback),
                        'label':'SPSA(100,cb)' }
    optimizers['16'] = {'opt': SPSA(maxiter=500,callback=SPSA_callback),
                        'label':'SPSA500,cb)' }
    return optimizers

