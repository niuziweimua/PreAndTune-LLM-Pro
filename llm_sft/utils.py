import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union, Dict

import openai
import tqdm
# from openai import openai_object
# note that after 1107, the openai package seems to be updated to openai.api_client
import copy

StrOrOpenAIObject = Union[str]

openai_org = os.getenv("OPENAI_ORG")
if openai_org is not None:
    openai.organization = openai_org
    logging.warning(f"Switching to organization: {openai_org} for OAI API key.")


@dataclasses.dataclass
class OpenAIDecodingArguments(object):
    max_to