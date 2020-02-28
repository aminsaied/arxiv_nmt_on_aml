"""Sample code invoking the webservice.
"""
import requests
import json

headers = {'Content-Type': 'application/json'}

if service.auth_enabled:
    headers['Authorization'] = 'Bearer '+service.get_keys()[0]
elif service.token_auth_enabled:
    headers['Authorization'] = 'Bearer '+service.get_token()[0]

print(headers)

test_abstract = """
The groups $\gamma_{n,s}$ are defined in terms of homotopy equivalences of certain graphs, and are natural generalisations of $Out(Fn)$ and $Aut(Fn)$. They have appeared frequently in the study of free group automorphisms, for example in proofs of homological stability in [8,9] and in the proof that $Out(Fn)$ is a virtual duality group in [1]. More recently, in [5], their cohomology $H_i(\Gamma_{n,s})$, over a field of characteristic zero, was computed in ranks $n=1,2$ giving new constructions of unstable homology classes of $Out(Fn)$ and $Aut(Fn)$. In this paper we show that, for fixed $i$ and $n$, this cohomology $H_iGammans$ forms a finitely generated FI-module of stability degree $n$ and weight $i$, as defined by Church-Ellenberg-Farb in [2].
"""

test_sample = json.dumps({'data': ''})

response = requests.post(
    service.scoring_uri, data=test_sample, headers=headers)
print(response.status_code)
print(response.elapsed)
print(response.json())