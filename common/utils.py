import gc
import os
import sys
import json
import random
import functools
from os import PathLike
from pathlib import Path
from datetime import datetime
from dataclasses import fields, asdict, _MISSING_TYPE, is_dataclass, Field
import typing as ty
from typing import TypeVar, Union, Callable
from argparse import ArgumentParser, Namespace

import yaml
import torch
import numpy as npy

__all__ = [
    '_gc',
    'json_dump_dict',
    'json_load_dict',
    'dict_dump_dataclass',
    'json_dump_dataclass',
    'json_load_dataclass',
    'parse_comma_list_int',
    'parse_comma_list_float',
    'get_datacls_parser',
    'dataclass_parser',
    'parse_dataclass_args',
    'dataclasses_argument_parse',
    'recursive_rm_dir',
    'record_runtime',
    'pass_kargs_to_sys_argv',
    'yaml_dump_dict',
    'yaml_load_dict',
    'get_dict',
]


def _gc():
    gc.collect()
    torch.cuda.empty_cache()

def set_random_seed_np_torch(seed: int, set_deterministic = False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    npy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if set_deterministic: # NOTE: better not, it takes much more time for F.linear
        # for cuda>10.1 the cublas for F.linear is not deterministic
        # it would take more than double time for deterministic cublas
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        # :16:8 is another choice with more time cost
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True)
        elif hasattr(torch, "set_deterministic"):
            # `set_deterministic` is new in torch 1.7.0
            torch.set_deterministic(True)

def json_dump_dict(data: dict, file_path: PathLike, **kwargs) -> None:
    if len(kwargs) == 0:
        kwargs = { 'indent': 4, 'ensure_ascii': False }
    for k, v in data.items():
        if isinstance(v, Path):
            data[k] = str(v)
    with open(file_path, 'w', encoding='UTF-8') as f:
        json.dump(data, f, **kwargs)


def json_load_dict(file_path: PathLike) -> dict:
    with open(file_path, 'r', encoding="UTF-8") as f:
        return json.load(f)
    

DataclassCls = TypeVar('DataclassCls')


def dict_dump_dataclass(cls: type[DataclassCls], data: dict, 
                        keys_set: set = None) -> DataclassCls:
    if keys_set is None:
        keys_set = set(data.keys())
    field_names = { field.name for field in fields(cls) }
    sub_keys = keys_set & field_names
    # here we receive the and set
    data_cls_data = { key: data[key] for key in sub_keys }
    return cls(**data_cls_data)

 
def json_dump_dataclass(data: DataclassCls, file_path: PathLike, **kwargs) -> None:
    data_dict = asdict(data)
    json_dump_dict(data_dict, file_path, **kwargs)


def json_load_dataclass(file_path: PathLike, dataclass_cls: type[DataclassCls]) -> DataclassCls:
    data = json_load_dict(file_path)
    return dict_dump_dataclass(dataclass_cls, data)


def parse_comma_list_int(comma_list: str) -> list[str]:
    texts = comma_list.split(',')
    return [int(text) for text in texts]


def parse_comma_list_float(comma_list: str) -> list[float]:
    texts = comma_list.split(',')
    return [float(text) for text in texts]


def parse_bool_str(bool_str: str) -> bool:
    if bool_str.lower() == 'true':
        return True
    elif bool_str.lower() == 'false':
        return False
    else:
        raise ValueError(f'bool_str should be "True" or "False", but got {bool_str}')


def get_cmd_arg_info(arg: Field, arg_type = None)\
        -> tuple[Callable, bool, ty.Optional[ty.Any], str]:
    help_msg = arg.metadata.get('help', '')
    has_default = not isinstance(arg.default, _MISSING_TYPE)
    default = None

    if has_default:
        help_msg = (f'{help_msg}, default: {arg.default}' if help_msg
                    else f'default: {arg.default}')
        default = arg.default
    
    if 'required' in arg.metadata:
        required = arg.metadata['required']
    else:
        required = not has_default

    if arg_type is None:
        arg_type = arg.type

    # special case for certain types
    if arg_type == bool:
        return parse_bool_str, required, default, help_msg
    
    if ty.get_origin(arg_type) is ty.Literal:
        choices = ty.get_args(arg_type)
        def literal_choice_parse_f(arg_choice: str):
            for choice in choices:
                typed_pass_arg = type(choice)(arg_choice)
                if typed_pass_arg == choice:
                    return typed_pass_arg
            raise ValueError(f'arg_choice {arg_choice} not in literal{choices}')
        
        return literal_choice_parse_f, required, default, help_msg

    if ty.get_origin(arg_type) is ty.Union:  # if is optional
        if  ty.get_args(arg_type)[1] is type(None):
            arg_type = ty.get_args(arg_type)[0]
            return get_cmd_arg_info(arg, arg_type)

    return arg.type, required, default, help_msg


def get_datacls_parser(dataclass_cls, parser = None, short_names: dict[str, str] = None):
    # NOTE: if don't have default, set it required, but if there is any better solution?
    if parser is None:
        parser = ArgumentParser()
    for a in fields(dataclass_cls):
        names = [f'--{a.name}']
        short_name = short_names.get(a.name, None)
        if short_name:
            names = [f'-{short_name}'] + names

        type_func, required, default, help_msg = get_cmd_arg_info(a)
            
        parser.add_argument(*names, type=type_func, default=default, required=required,
                            help=help_msg, choices=a.metadata.get('choices', None))
    return parser


def dataclass_parser(*dataclass_cls):
    parser = ArgumentParser()
    short_name_set = { 'h' }
    short_names = {}
    for cls in dataclass_cls:
        for a in fields(cls):
            if a.name not in short_name_set:
                short_name = ''.join([ s[0] for s in a.name.split('_')])
                for i in range(1, len(short_name)+1):
                    sn = short_name[:i]
                    if sn not in short_name_set:
                        short_name_set.add(sn)
                        short_names[a.name] = sn
                        break
    
    for cls in dataclass_cls:
        get_datacls_parser(cls, parser, short_names)
    return parser


_DCls = TypeVar('_DCls')
_DCls1 = TypeVar('_DCls1')
_DCls2 = TypeVar('_DCls2')
_DCls3 = TypeVar('_DCls3')
_DCls4 = TypeVar('_DCls4')
_DCls5 = TypeVar('_DCls5')


@ty.overload
def parse_dataclass_args(args: Union[Namespace, dict], dcls1: ty.Type[_DCls]
                         ) -> _DCls:
    ...


@ty.overload
def parse_dataclass_args(args: Union[Namespace, dict], dcls1: ty.Type[_DCls1], 
                         dcls2: ty.Type[_DCls2]) -> tuple[_DCls1, _DCls2]:
    ...

@ty.overload
def parse_dataclass_args(args: Union[Namespace, dict], dcls1: ty.Type[_DCls1], 
                         dcls2: ty.Type[_DCls2], dcls3: ty.Type[_DCls3]
                         ) -> tuple[_DCls1, _DCls2, _DCls3]:
    ...

@ty.overload
def parse_dataclass_args(args: Union[Namespace, dict], dcls1: ty.Type[_DCls1], 
                         dcls2: ty.Type[_DCls2], dcls3: ty.Type[_DCls3], 
                         dcls4: ty.Type[_DCls4]) -> tuple[_DCls1, _DCls2, _DCls3, _DCls4]:
    ...

@ty.overload
def parse_dataclass_args(args: Union[Namespace, dict], dcls1: ty.Type[_DCls1], 
                         dcls2: ty.Type[_DCls2], dcls3: ty.Type[_DCls3], 
                         dcls4: ty.Type[_DCls4], dcls5: ty.Type[_DCls5])\
                         -> tuple[_DCls1, _DCls2, _DCls3, _DCls4]:
    ...

def parse_dataclass_args(args: Union[Namespace, dict], *dataclass_cls: ty.Type[_DCls]
                         )-> tuple[_DCls, ...]:
    if isinstance(args, Namespace):
        parsed_dict = args.__dict__
    elif isinstance(args, dict):
        parsed_dict = args
    else:
        raise TypeError(f'args should be Namespace or dict, but got {type(args).__name__}')
    parse_result = []
    keys = set(parsed_dict.keys())
    for cls in dataclass_cls:
        parse_result.append(dict_dump_dataclass(cls, parsed_dict, keys))
    if len(parse_result) == 1:
        return parse_result[0]
    return parse_result


@ty.overload
def dataclasses_argument_parse(dcls1: ty.Type[_DCls1], args_seq=None) -> _DCls1:
    ...

@ty.overload
def dataclasses_argument_parse(dcls1: ty.Type[_DCls1], dcls2: ty.Type[_DCls2],
                               args_seq=None) -> tuple[_DCls1, _DCls2]:
    ...

@ty.overload
def dataclasses_argument_parse(dcls1: ty.Type[_DCls1], dcls2: ty.Type[_DCls2],
                               dcls3: ty.Type[_DCls3], args_seq=None
                               ) -> tuple[_DCls1, _DCls2, _DCls3]:
    ...

@ty.overload
def dataclasses_argument_parse(dcls1: ty.Type[_DCls1], dcls2: ty.Type[_DCls2],
                               dcls3: ty.Type[_DCls3], dcls4: ty.Type[_DCls4],
                               args_seq=None) -> tuple[_DCls1, _DCls2, _DCls3, _DCls4]:
    ...

@ty.overload
def dataclasses_argument_parse(dcls1: ty.Type[_DCls1], dcls2: ty.Type[_DCls2], 
                               dcls3: ty.Type[_DCls3], dcls4: ty.Type[_DCls4],
                               dcls5: ty.Type[_DCls5], args_seq=None)\
                                -> tuple[_DCls1, _DCls2, _DCls3, _DCls4, _DCls5]: 
    ...

@ty.overload
def dataclasses_argument_parse(*dataclass_cls: ty.Type[_DCls],
                               args_seq=None) -> tuple[_DCls, ...]:
    ...

def dataclasses_argument_parse(*dataclass_cls, args_seq=None):
    parser = dataclass_parser(*dataclass_cls)
    args = parser.parse_args(args_seq)
    r = parse_dataclass_args(args, *dataclass_cls)
    if len(dataclass_cls) == 1:
        # NOTE: parse_dataclass_args returns the dataclass instance instead of list
        return r
    return r

def recursive_rm_dir(dir_path: Union[PathLike, str]):
    if not isinstance(dir_path, Path):
        dir_path = Path(dir_path)
    if dir_path.is_file():
        dir_path.unlink()
    else:
        for file_or_dir in dir_path.glob('*'):
            recursive_rm_dir(file_or_dir)
        dir_path.rmdir()


def dump_name_idx_txt(name2idx: dict[str, int], file_path: PathLike):
    with open(file_path, 'w', encoding='UTF-8') as f:
        for name, idx in name2idx.items():
            f.write(f'{name}\t{idx}\n')


def read_name_idx_txt(file_path: PathLike) -> dict[str, int]:
    name_idx_map = dict()
    with open(file_path, 'r', encoding='UTF-8') as f:
        for line in f:
            item, idx = line.strip().split('\t')
            name_idx_map[item] = int(idx)
    return name_idx_map


def datacls_format_str(data): 
    if not is_dataclass(data):
        raise ValueError("data should be dataclass")
    data_dict = asdict(data)
    max_key_len = max([len(key) for key in data_dict.keys()])
    parsed_str = type(data).__name__ + ':\n'
    for key, value in data_dict.items():
        key = ' ' * (max_key_len-len(key)) + key
        parsed_str += f'\t{key} : {value}\n'
    return parsed_str


class TimeRecoder:
    def __init__(self, describ: str = '', log_func: ty.Callable[[str], None] = None):
        self.describ = describ
        self.start = None
        self.run_time = None
        self.log_func = log_func or print

    def __enter__(self):
        self.start = datetime.now()
    
    def __exit__(self, e, ev, cb):
        self.run_time = datetime.now()-self.start
        self.log_func(f'{self.describ} runtime: {self.run_time}')

    def __call__(self, func):
        assert callable(func), "Time Recorder only accept callable objects "\
                              f"but recieved {type(func).__name__}."
        @functools.wraps(func)
        def wrapper(*args, **kargs):
            with self:
                return func(*args, **kargs)
        return wrapper


def record_runtime(func_or_descb: Union[Callable, str, None] = None):
    if callable(func_or_descb):
        return TimeRecoder(func_or_descb.__name__)(func_or_descb)
    
    describ = func_or_descb or ''

    return TimeRecoder(describ)


def pass_kargs_to_sys_argv(kargs: dict):
    if len(sys.argv) > 1:
        # print(f"{[type(self).__name__]} Warning: sys.argv is not empty: {sys.argv[1:]}")
        sys.argv = sys.argv[:1]
    
    for k, v in kargs.items():
        # consider there might be short names and cmd args can't begin with '-'
        if k.startswith('-'):
            sys.argv.append(k)
        else:
            sys.argv.append(f"--{k}")
        sys.argv.append(str(v))


class YamlDumper(yaml.Dumper):
    def increase_indent(self, flow=True, indentless=False):
        return super().increase_indent(flow, indentless)
    

def yaml_dump_dict(data: dict, file_path: PathLike, **kwargs) -> None:
    with open(file_path, 'w', encoding='UTF-8') as f:
        yaml.dump(data, f, Dumper=YamlDumper, **kwargs)


def yaml_load_dict(file_path: PathLike) -> dict:
    with open(file_path, 'r', encoding='UTF-8') as f:
        return yaml.load(f, Loader=yaml.Loader)


def get_dict(data: dict, *keys: list[str],
             **default) -> tuple[ty.Optional[ty.Any], ...]:
    return tuple(data.get(key, default.get(key, None)) for key in keys)
