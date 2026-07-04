import asyncio
import os
import json
import functools
import time
import openai

class UnRetryException(BaseException):
    def __init__(self, message=""):
        self.message = message
        super().__init__(self.message)

# def retry(max_retries, time_sleep):
#     def decorator_retry(func):
#         @functools.wraps(func)
#         def wrapper_retry(*args, **kwargs):
#             attempts = 0
#             while attempts < max_retries:
#                 try:
#                     return func(*args, **kwargs)
#                 except openai.error.APIError as e:
#                     # import pdb; pdb.set_trace()
#                     error_json = json.loads(e.body)
#                     if error_json['code'] == 'context_length_exceeded':
#                         raise UnRetryException(f"API error: {e}")
#                     raise ValueError(f"API error: {e}")
#                 except openai.error.InvalidRequestError as e:
#                     raise ValueError(f"Invalid request error: {e}")
#                 except UnRetryException as e:
#                     raise ValueError(f"UnRetryException: {e}")
#                 except Exception as e:
#                     attempts += 1
#                     if attempts >= max_retries:
#                         print(f"All {max_retries} attempts failed.")
#                         import traceback
#                         traceback.print_exc()
#                         raise ValueError(e)
#                     time.sleep(time_sleep)
#         return wrapper_retry
#     return decorator_retry

def retry(max_retries, time_sleep, tag="default"):
    def decorator_retry(func):
        @functools.wraps(func)
        async def async_wrapper_retry(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return await func(*args, **kwargs)
                except UnRetryException as e:
                    raise ValueError(f"UnRetryException: {e}")
                except Exception as e:
                    attempts += 1
                    import traceback
                    traceback.print_exc()
                    print(f"{tag}: Attempt {attempts} failed: {e}")
                    if attempts >= max_retries:
                        print(f"{tag}: All {max_retries} attempts failed")
                        raise ValueError(f"{tag}: All {max_retries} attempts failed")
                    await asyncio.sleep(time_sleep)  # 使用异步睡眠

        @functools.wraps(func)
        def sync_wrapper_retry(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except UnRetryException as e:
                    raise ValueError(f"UnRetryException: {e}")
                except Exception as e:
                    attempts += 1
                    import traceback
                    traceback.print_exc()
                    print(f"{tag}: Attempt {attempts} failed: {e}")
                    if attempts >= max_retries:
                        print(f"{tag}: All {max_retries} attempts failed")
                        raise ValueError(f"{tag}: All {max_retries} attempts failed")
                    time.sleep(time_sleep)

        # 根据函数是否是协程函数返回对应的包装器
        if asyncio.iscoroutinefunction(func):
            return async_wrapper_retry
        return sync_wrapper_retry

    return decorator_retry