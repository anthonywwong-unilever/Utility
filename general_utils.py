from typing import List, Dict, Callable, Any, Union


def list_mapping(lst: List[Any], transform: Callable) -> Dict[Any, Any]:
    """Return a dictionary where each key is an element from <lst>, and
    its corresponding value is the element being mapped to under <transform>.
    """
    to_lst = [transform(element) for element in lst]
    return {from_element: to_element for from_element, to_element in zip(lst, to_lst)}


def clean_columns(columns: List[str]) -> Dict[str, str]:
    clean_up = lambda col: '_'.join(col.lower().strip().split(' '))
    return list_mapping(columns, clean_up)


def lists_to_dict(lst_1: List[Any], lst_2: List[Any]) -> Dict[Any, Any]:
    """Return a dictionary where each key-value pair is the pair-wise elements
    from <lst_1> and <lst_2>, respectively.
    """
    assert len(lst_1) == len(lst_2)
    return {key: value for key, value in zip(lst_1, lst_2)}


def filter_list(lst: List[Any], conditions: Callable) -> List[Any]:
    """Filter for elements in <lst> that satisfies the set of <conditions> in the order
    each condition is laid out.
    """
    # if isinstance(conditions, Callable):
    return list(filter(conditions, lst))
    # else:
    #     first_filtered = list(filter(conditions[0], lst))
    #     next_conditions = conditions[1:] if len(conditions[1:]) > 1 else conditions[-1]
    #     return filter_list(first_filtered, next_conditions)
    

def filter_contain(lst: List[Any], lst_contain: List[Any]) -> List[Any]:
    contain = lambda lst_element: lst_element in lst_contain
    return filter_list(lst, contain)


def filter_contain_str(lst: List[str], word: str) -> List[str]:
    contain = lambda lst_element: word in lst_element
    return filter_list(lst, contain)


def filter_not_contain(lst: List[Any], lst_contain: List[Any]) -> List[Any]:
    not_contain = lambda lst_element: lst_element not in lst_contain 
    return filter_list(lst, not_contain)


def filter_not_contain_str(lst: List[str], words: Union[str, List[str]]) -> List[str]:
    if isinstance(words, str):
        not_contain = lambda lst_element: words not in lst_element
        return filter_list(lst, not_contain)
    no_first_word_lst = filter_list(lst, lambda lst_element: words[0] not in lst_element)
    return filter_list(no_first_word_lst, words[1:]) if len(words[1:]) > 1 else filter_list(no_first_word_lst, words[0])
    

def filter_startswith(lst: List[str], element: str) -> List[str]:
    startswith = lambda lst_element: lst_element.startswith(element)
    return filter_list(lst, startswith)


def filter_not_startswith(lst: List[str], element: str) -> List[str]:
    not_startswith = lambda lst_element: not lst_element.startswith(element)
    return filter_list(lst, not_startswith)


def filter_endswith(lst: List[str], element: str) -> List[str]:
    endswith = lambda lst_element: lst_element.endswith(element)
    return filter_list(lst, endswith)


def filter_not_endswith(lst: List[str], element: str) -> List[str]:
    not_endswith = lambda lst_element: not lst_element.endswith(element)
    return filter_list(lst, not_endswith)


def filter_greater(lst: List[Union[int, float]], number: Union[int, float]) -> List[Union[int, float]]:
    greater_than = lambda lst_number: lst_number > number
    return filter_list(lst, greater_than)


def filter_equal(lst: List[Union[int, float]], number: Union[int, float]) -> List[Union[int, float]]:
    equal_to = lambda lst_number: lst_number == number
    return filter_list(lst, equal_to)


def filter_smaller(lst: List[Union[int, float]], number: Union[int, float]) -> List[Union[int, float]]:
    smaller_than = lambda lst_number: lst_number < number
    return filter_list(lst, smaller_than)


def filter_greater_equal(lst: List[Union[int, float]], number: Union[int, float]) -> List[Union[int, float]]:
    greater, equal  = filter_greater(lst, number), filter_equal(lst, number)
    return equal + greater


def filter_smaller_equal(lst: List[Union[int, float]], number: Union[int, float]) -> List[Union[int, float]]:
    smaller, equal = filter_smaller(lst, number), filter_equal(lst, number)
    return smaller + equal

