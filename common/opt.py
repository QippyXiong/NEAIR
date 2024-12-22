import os
import sys
import math
import shutil
import traceback
import importlib
import typing as ty
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass
from argparse import ArgumentParser

import yaml
from torch.cuda import OutOfMemoryError

import optuna

from . import utils
from .utils import _gc, pass_kargs_to_sys_argv, yaml_load_dict, yaml_dump_dict,\
    json_load_dict, json_dump_dict


def get_objective(entry, use_grad_acc, skip_all_except, 
                  except_cb: ty.Callable[[Exception], ty.Any] = None,
                  still_oom_cb: ty.Callable[[int, int], ty.Any] = None,
                  ) -> ty.Callable[[dict], ty.Any]:
    def default_except_cb(e):
        traceback.print_exc()
        return float('-inf')
    
    def default_still_oom_cb(grad_acc_step, batch_size):
        print(f"cuda still oom with batch size {batch_size} and acc step {grad_acc_step}.")
        traceback.print_exc()
        return float('-inf')

    except_cb = except_cb or default_except_cb
    still_oom_cb = still_oom_cb or default_still_oom_cb
    
    if not use_grad_acc:
        def objective(kargs):
            pass_kargs_to_sys_argv(kargs)
            try:
                return entry()
            except Exception as e:
                if skip_all_except:
                    fail_value = except_cb(e)
                    return fail_value
                raise
        return objective

    def objective_with_grad_acc(kargs: dict):
        grad_acc_step = kargs.get('grad_acc_step', 1)
        grad_acc_step = kargs.get('--grad_acc_step', grad_acc_step)
        grad_acc_step = int(grad_acc_step)
        if 'batch_size' not in kargs:
            assert '--batch_size' in kargs, "batch_size not in kargs."
        batch_size = kargs.get('batch_size', kargs.get('--batch_size', None))
        batch_size = int(batch_size)

        while grad_acc_step <= batch_size:
            kargs['grad_acc_step'] = grad_acc_step
            pass_kargs_to_sys_argv(kargs)
            try:
                return entry()
            except Exception as e:
                if (isinstance(e, OutOfMemoryError)
                    or 'CUDA out of memory' in str(e)):
                    grad_acc_step += 1
                    while (batch_size % grad_acc_step != 0 
                            and grad_acc_step <= batch_size):
                        grad_acc_step += 1

                    if grad_acc_step > batch_size:
                        v = still_oom_cb(grad_acc_step, batch_size)
                        return v
                    print(f"drop grad_acc_step to {grad_acc_step} for "
                          f"batch_size={batch_size}")
                    continue
                if skip_all_except:
                    return except_cb(e)
                raise
    return objective_with_grad_acc


class CmdArgOptStudy:
    studies: ty.Dict[str, optuna.study.Study]

    optuna_config_keys = {'n_trials', 'timeout', 'n_jobs', 'same_as'}
    default_direction = 'maximize'

    @ty.overload
    def __init__(self, config_path: os.PathLike, save_path=None):
        ...

    @ty.overload
    def __init__(self, config: dict, save_path=None):
        ...

    def __init__(self, config_or_path, save_path=None, skip_all_except=True, use_grad_acc=True):
        if isinstance(config_or_path, os.PathLike) or isinstance(config_or_path, str):
            with open(config_or_path, 'r', encoding='utf-8') as fp:
                config = yaml.safe_load(fp)
        elif isinstance(config_or_path, dict):
            config = config_or_path
        else:
            raise ValueError(f"Unsupport type {type(config_or_path).__name__} for config.")

        self.config = config
        self.save_path = Path(save_path)
        self.skip_all_except = skip_all_except
        self.use_grad_acc = use_grad_acc

        if self.save_path:
            # NOTE: config dumped here would be checked for "same as" statement in following
            os.makedirs(self.save_path, exist_ok=True)
            # store yaml here
            yaml_dump_dict(config, self.save_path/'config.yaml')

        self.studies = None

        # check run_studies, use the set run_studies for selected study for running
        self.run_studies = self.config.pop('run_studies', list(self.config.keys()))

        # check "same as" statement
        for study_name in self.config:
            if "same_as" in self.config[study_name]:
                refer_name = self.config[study_name]["same_as"]
                ori_config = self.config[study_name]
                self.config[study_name] = deepcopy(self.config[refer_name])
                self.config[study_name].update(ori_config)

        self.error_log_file = None
 
    
    def suggest_config(self, trial: optuna.Trial, study_name):
        trail_suggest = dict()
        for k, v in self.config[study_name].items():
            # maybe 
            suggested_value = None
            if k in self.optuna_config_keys:
                continue
            if isinstance(v, dict):
                suggested_value = self.add_dict_trial(trial, k, v)
            elif isinstance(v, list):
                # Literal
                suggested_value = self.add_literal_trail(trial, k, v)
            else:
                print(f"[Optuna Warning] Unkown type{type(v).__name__} for key {k} optuna search"
                      ", skip.")
            if suggested_value is not None:
                trail_suggest[k] = suggested_value
        return trail_suggest

    def add_dict_trial(self, trail: optuna.Trial, confk: str, confv: dict):
        try:
            arg_type = confv['type'].lower()
            suggested_value = None
            if arg_type == 'literal' or arg_type == 'str':
                suggested_value = self.add_literal_trail(trail, confk, confv['value'])
            if arg_type == 'int' or arg_type == 'float':
                func = getattr(trail, f"suggest_{arg_type}")
                arg_type = int if arg_type == 'int' else float

                step = None
                if 'low' in confv and 'high' in confv:
                    low, high = confv['low'], confv['high']
                    step = arg_type(confv['step'])
                elif 'range' in confv:
                    v = confv['range']
                    if len(v) == 2:
                        low, high = v
                        step = confv.get('step', None)
                    elif len(v) == 3:
                        low, high, step = v
                    else:
                        raise ValueError(f"Unsupport range length {len(v)} for optuna search "
                                         f"of config key {confk}")

                    low, high = arg_type(low), arg_type(high)
                    if step is None:
                        suggested_value = func(confk, low, high)
                    else:
                        step = arg_type(step)
                        suggested_value = func(confk, low, high, step=step)
                elif 'value' in confv:
                    values = [ arg_type(d) for d in confv['value']]
                    suggested_value = trail.suggest_categorical(confk, values)
                else:
                    raise ValueError(f"missing key 'range', 'low' and 'high' or 'value' in optuna "
                                     f"search config key {confk}.")
            else:
                raise ValueError(f"Unsupport type {arg_type} of {confk} for optuna search.")
            assert suggested_value is not None, "suggested value should not be None."
            return suggested_value

        except KeyError as e:
            # if default using key is missing
            if e.args[0] in { 'type' }:
                raise KeyError(f"missing default key {e.args[0]} for optuna search. ") from e
            raise

    def add_literal_trail(self, trail: optuna.Trial, confk: str, confv: list):
        return trail.suggest_categorical(confk, confv)

    def create_studies_maybe(self):
        if self.studies is None:
            self.studies = dict()
        
        for study_name, conf in self.config.items():
            db_path = f"sqlite:///{self.save_path}/trials.db" if self.save_path else ...
            if study_name not in self.studies:
                self.studies[study_name] = optuna.create_study(
                    study_name=study_name, storage=db_path, 
                    direction=conf.get('direction', self.default_direction),
                    load_if_exists=True)

    def open_error_log_file_maybe(self):
        if self.save_path and self.skip_all_except:
            self.error_log_file = open(f"{self.save_path}/error.log", 'w', encoding='utf-8')
        
    def get_objective(self, entry, study_name, extra_kargs = None):

        def except_cb(e):
            _gc()
            traceback.print_exc(file=self.error_log_file or sys.stderr)
            self.error_log_file.flush()
            return float('-inf')

        def still_oom_cb(grad_acc_step, batch_size):
            print(f"{type(self).__name__} cuda still oom with batch size "
                    f"{batch_size} and acc step {grad_acc_step}.")
            traceback.print_exc(file=self.error_log_file or sys.stderr)
            self.error_log_file.flush()
            return float('-inf')

        main = get_objective(entry, self.use_grad_acc, self.skip_all_except,
                             except_cb, still_oom_cb)

        def objective(trial):
            kargs = self.suggest_config(trial, study_name)
            if extra_kargs:
                kargs.update(extra_kargs)
            r = main(kargs)

            try:  # save other metrics to optuna trial if have
                other_metrics_dict = r.metric_dict()
                for k, v in other_metrics_dict.items():
                    trial.set_user_attr(k, v)
            except (AttributeError, NotImplementedError):
                pass
            r_value = float(r)
            del r   # avoid reference from entry and cause memory leak
            _gc()
            return r_value
        return objective

    def run_optuna(self, entry, study_names = None, extra_kargs = None, **opt_kwargs) -> None:
        if isinstance(study_names, str):
            study_names = [study_names]  
        if study_names is None:
            study_names = self.run_studies

        self.create_studies_maybe()
        self.open_error_log_file_maybe()
        for study_name in study_names:
            if 'n_trials' in self.config[study_name] and 'n_trials' not in opt_kwargs:
                opt_kwargs['n_trials'] = int(self.config[study_name]['n_trials'])
            if 'timeout' in self.config[study_name] and 'timeout' not in opt_kwargs:
                opt_kwargs['timeout'] = float(self.config[study_name]['timeout'])
            if 'n_jobs' in self.config[study_name] and 'n_jobs' not in opt_kwargs:
                opt_kwargs['n_jobs'] = int(self.config[study_name]['n_jobs'])

            self.studies[study_name].optimize(self.get_objective(entry, study_name, extra_kargs),
                                              gc_after_trial=True, **opt_kwargs)
        if self.error_log_file:
            self.error_log_file.close()

    def load_study_trials(self, save_db_file_path):
        self.studies = {
            study_name: optuna.load_study(study_name=study_name, 
                                          storage=f"sqlite:///{save_db_file_path}")
            for study_name in self.config
        }
    
    def save(self):
        raise NotImplementedError("If you really want to save the study, you should "
                                  "give the save_path in __init__")

    @classmethod
    def load(cls, save_dir_path):
        path = Path(save_dir_path)
        with open(path/'config.yaml', 'r', encoding='utf-8') as fp:
            config = yaml.safe_load(fp)
        study = cls(config, save_path=save_dir_path)
        study.load_study_trials(path/'trials.db')
        return study
    
    def export_best_trial(self, export_bash_path = None, entry_name="<entry.py>"):
        export_bash_content = "# auto generated bash script, add the actualy run "\
                              "command to the end of this script\n\n"
        
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        RESET = "\033[0m"
        try:
            os.get_terminal_size()
        except OSError:
            RED     = ""
            GREEN   = ""
            YELLOW  = ""
            BLUE    = ""
            MAGENTA = ""
            CYAN    = ""
            RESET   = ""

        N_columns = shutil.get_terminal_size().columns
        cmd_keys = None
        config_for_bash = dict()
        for name, optuna_study in self.studies.items():
            try:
                trail = optuna_study.best_trial
            except ValueError as e:
                if "Record does not exist" in str(e):
                    print(f"{GREEN}Study {name}:\n{RESET}"
                          +"="*int(N_columns * 0.8) + "\n"
                          +"No Records\n")
                    continue
            cmd_log = ""
            cmd_log += f"{GREEN}Study {name}{RESET}\n"
            cmd_log += "="*int(N_columns * 0.8)+"\n"
            cmd_log += f"{MAGENTA}Best main value: {trail.values[0]}{RESET}\n"
            if trail.user_attrs:
                cmd_log += f"{CYAN}Best other attrs:{RESET}\n"
                cmd_log += "\n".join([f"    {BLUE}{k}{RESET}: {v}" 
                                      for k, v in trail.user_attrs.items()])
                cmd_log += "\n"

            cmd_log += f"{YELLOW}Best params:{RESET}\n"
            cmd_log += "\n".join([f"    {BLUE}{k}{RESET}: {v}" 
                                  for k, v in trail.params.items()])
            cmd_log += "\n"
            print(cmd_log)

            if cmd_keys is None:
                cmd_keys = set(trail.params.keys())
            else:
                extra_keys = set(trail.params.keys()) - cmd_keys
                # little tricky if cmd kargs not same ...
                # TODO: take the whole cmd keys for generated bash
                if extra_keys:
                    print("[Warning] cmd kargs not same for different study, Maybe "
                          "not export bash script?")

            config_for_bash[name] = trail.params
        
        if export_bash_path:
            for name, params in config_for_bash.items():
                export_bash_content += f'if [ "$1" = "{name}" ]; then\n'
                for k, v in params.items():
                    # if isinstance(v, float):
                    #     v = f"{v:.4f}" if 1e3 > v > 1e-3\
                    #         else f"{v:.3e}"
                    if isinstance(v, str):
                        v = f'"{v}"'
                    elif isinstance(v, bool):
                        v = str(v)
                    export_bash_content += f"    {k}={v}\n"
                export_bash_content += "fi\n\n"

            export_bash_content += "\n\n"

            export_bash_content += f"python {entry_name}\\\n"
            for key in cmd_keys:
                export_bash_content += f"    --{key} ${key}\\\n"
            export_bash_content += "    \"${@:2}\""
            with open(export_bash_path, 'w', encoding='utf-8') as fp:
                fp.write(export_bash_content)


@dataclass
class GridSearchStatus:
    cur_study_idx: int = -1
    cur_grid_idx: int = -1
    total_grid_num: int = -1

    run_studies: list[str] = None
    @property
    def total_study_num(self):
        return len(self.run_studies)

    cur_study_name: str = ""
    # not including default args for the study
    cur_grid_conf: ty.Optional[ty.Dict[str, ty.Any]] = None


class CmdArgGridSearchStudy:
    r"""
    config: 
    ``` 
    - study_name: { arg_key: [arg_value, ...]},
    ...
    - default_args: { study_name: {arg_key: [arg_value, ...]} }
    - run_studies: [study_name, ...]
    ```
    default_args provide the default args for each study
    """

    GRID_CONFIG_FILE_NAME = 'grid_config.yaml'
    STATUS_FILE_NAME = 'status.json'
    DEFAULT_CONFIG_KEYS = { 'default_args', 'run_studies' }

    status: GridSearchStatus
    config: dict[str, dict]

    def __init__(self, config_or_config_path = None, 
                 save_dir = None, force_load = False, **kwargs):
        # if kwargs not invalid ...
        assert (config_or_config_path or save_dir) is not None, \
                "cannot recieve both None for config and save_dir"
        
        # pop kwargs
        status = kwargs.pop('status', GridSearchStatus())
        results = kwargs.pop('results', {})

        assert len(kwargs)==0, f"unkown containing extra kwargs {list(kwargs.keys())}"

        self.save_dir = None
        if save_dir is not None:
            self.save_dir = Path(save_dir)
        
        load_need = (save_dir is not None
                    and config_or_config_path is None
                    and self.save_dir.exists()) or force_load

        if load_need:
            assert self.save_dir, "should give save_dir for loading."
            assert self.save_dir.exists(), f"save_dir {self.save_dir} not exists."
            # load config
            config_or_config_path = yaml_load_dict(self.save_dir/
                                                   self.GRID_CONFIG_FILE_NAME)
            status = utils.json_load_dataclass(self.save_dir/self.STATUS_FILE_NAME,
                                               GridSearchStatus)
            results = json_load_dict(self.save_dir/'results.json')

        # if save_dir has results and not set pass results, just load the saved
        if len(results) == 0 and (self.save_dir/'results.json').exists():
            results = json_load_dict(self.save_dir/'results.json')

        if isinstance(config_or_config_path, os.PathLike):
            config = yaml_load_dict(config_or_config_path)
        elif isinstance(config_or_config_path, dict):
            config = config_or_config_path
        else:
            raise ValueError(f"Unsupport type {type(config_or_config_path).__name__}"
                              " for config.")

        self.status = status
        self.config = config
        self.results = results

        self.use_grad_acc = kwargs.get('use_grad_acc', True)
        self.skip_all_except = kwargs.get('skip_all_except', True)

    default_args = property(lambda self: self.config.get('default_args', {}))


    @classmethod
    def load(cls, save_path):
        path = Path(save_path)
        config = yaml_load_dict(path/'grid_config.yaml')
        status = utils.json_load_dataclass(path/'status.json', GridSearchStatus)
        results = json_load_dict(path/'results.json')
        return cls(config, save_path, force_load=True,
                   status=status, results=results)


    @staticmethod
    def get_current_grid_args(idx, grid):
        grid_args = {}
        i = idx
        for k, v in grid.items():
            cur_idx = i % len(v)
            grid_args[k] = v[cur_idx]
            i = i // len(v)
        return grid_args
    
    def maybe_create_study_names(self, run_studies = None):
        if run_studies is not None:
            self.status.run_studies = run_studies
            for name in run_studies:
                assert name in self.config, f"running study name {name} not in config."
            return
        if self.status.run_studies is None:
            if 'run_studies' in self.config:
                self.status.run_studies = self.config['run_studies']
            else:
                self.status.run_studies = list(key for key in self.config 
                                                if key not in self.DEFAULT_CONFIG_KEYS)

    def step_increase(self):
        self.status.cur_grid_idx += 1
        
        if self.status.cur_grid_idx >= self.status.total_grid_num:
            self.status.cur_grid_idx = 0
            self.status.cur_study_idx += 1

        if self.status.cur_study_idx >= self.status.total_study_num:
            self.status.cur_grid_idx = self.status.total_grid_num
            return False

        if self.status.cur_grid_idx == 0:
            self.status.cur_study_name = list(self.config.keys())[self.status.cur_study_idx]
            self.status.total_grid_num = math.prod(len(v) for v in 
                                                self.config[self.status.cur_study_name].values())

        self.status.cur_grid_conf = self.get_current_grid_args(
            self.status.cur_grid_idx, self.config[self.status.cur_study_name])

        return True

    def parse_grid_key(self, k, v):
        return f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}"

    def grid_search(self, entry, study_names = None, extra_kargs = None):
        self.maybe_create_study_names(study_names)

        objective = get_objective(entry, self.use_grad_acc, self.skip_all_except)
        if self.save_dir is not None:
            self.save_dir.mkdir(exist_ok=True)
        self.save_config()
        while self.step_increase():
            args = extra_kargs or {}
            args.update(self.default_args.get(self.status.cur_study_name, {}))            
            args.update(self.status.cur_grid_conf)

            result = objective(args)

            result_key = '('+ ', '.join([ self.parse_grid_key(k,v)
                                          for k, v in self.status.cur_grid_conf.items() ]) + ')'
            self.results.setdefault(self.status.cur_study_name, {})
            self.results[self.status.cur_study_name][result_key] = str(result)
            del result
            _gc()
            self.save_status()
            self.save_results()

        self.save_all()

    def save_config(self):
        if self.save_dir is None:
            return
        yaml_dump_dict(self.config, self.save_dir/self.GRID_CONFIG_FILE_NAME)

    def save_status(self):
        if self.save_dir is None:
            return
        utils.json_dump_dataclass(self.status, self.save_dir/self.STATUS_FILE_NAME)
    
    def save_results(self):
        if self.save_dir is None:
            return
        json_dump_dict(self.results, self.save_dir/'results.json')
    
    def save_all(self):
        if self.save_dir is not None:
            self.save_dir.mkdir(exist_ok=True)
        self.save_config()
        self.save_status()
        self.save_results()


def default_main():
    parser = ArgumentParser(description="default optim entry for a quick optimization")
    parser.add_argument('--strategy', type=str, required=True,
                        help="strategy for optimization.")
    parser.add_argument('--entry_module', required=True,
                        help="entry python module for optimization.")
    parser.add_argument('--entry', default='main',
                        help="entry function for optimization.")
    parser.add_argument('--config_path', type=str,
                        help="config path for optimization.")
    parser.add_argument('--save_path', type=str,
                        help="save path for optimization.")
    parser.add_argument('--export', action='store_true', 
                        help="whether export the best trial to bash.")
    parser.add_argument('--bash_path', type=str, required=False,
                        help="bash path for exporting.")
    parser.add_argument('--save_path', type=str, required=False,
                        help="save path for optimization.")

    args, unkown = parser.parse_known_args()

    try:
        entry = getattr(importlib.import_module(args.entry_module), args.entry)
    except AttributeError:
        print(f"Cannot find {args.entry} in {args.entry_module}.", file=sys.stderr)
        return

    if args.export:
        study = CmdArgOptStudy.load(args.save_path)
        study.export_best_trial(args.bash_path)
        return
    
    config_path = args.config_path
    save_path = args.save_path
    # * TODO: actually this extra_kargs setting is not reasonable cause there may be
    # * cmd args which don't need a value.
    keys, values = unkown[::2], unkown[1::2]

    if args.strategy == 'optuna':
        study = CmdArgOptStudy(config_path, save_path=save_path)
        extra_kargs = dict(zip(keys, values))

        study.run_optuna(entry, extra_kargs=extra_kargs)
    elif args.strategy == 'grid_search':
        study = CmdArgGridSearchStudy(config_path, save_path=save_path)
        study.grid_search(entry)


if __name__ == '__main__':
    ...
