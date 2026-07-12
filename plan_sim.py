"""Copia local de src/datacol/plan_sim.py (hrc-data-collection), OS-7.

Nao altera o arquivo original em hrc-data-collection/. Unica mudanca: o
import de INTENTION_LIST aponta para a constante local (identica) em vez do
pacote datacol, para que este modulo funcione de forma independente dentro
de hrc-finetune/.

O modulo nao importa ROS nem executa movimentos. Ele reproduz apenas as
regras de decisao de ``decide_send_action`` e a atualizacao de estado que
ocorre apos uma acao ser concluida com sucesso.
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

INTENTION_LIST = {"no_action": 0, "get_connectors": 1, "get_screws": 2, "get_wheels": 3}

STAGES = (None, "bottom", "four_tubes", "top")
CONTEXT_DIMS = (7, 10)
PLAN_POLICIES = ("receiver", "proxy_graph")

_TUBE_LIMITS = {"short": 8, "long": 4}
_REVERT_TUBE = {
    "get_short_tubes": "long",
    "get_long_tubes": "short",
}
_ACTIONS = {
    "get_short_tubes",
    "get_long_tubes",
    "spin_bottom",
    "spin_four_tubes",
    "spin_top",
    "lift_up",
}
_COMMANDS = {"short", "long", "spin", "get up"}


class PlanGraph:
    """Estado e regras de transição do plano de montagem.

    Args:
        stageI_done: Inicializa o mesmo preset ``--stageI_done`` de
            ``controller/receiver.py``.
        policy: ``receiver`` replica as condições legadas; ``proxy_graph``
            preserva os dois ramos e limita cada estágio a quatro parafusos.

    A decisão e a conclusão da ação são operações separadas para preservar a
    semântica do controlador: ``apply_intention`` pode alterar contadores de
    parafusos/rodas, enquanto ``apply_action`` registra somente uma ação que
    terminou com sucesso.
    """

    def __init__(
        self,
        stageI_done: bool = False,
        policy: str = "receiver",
    ) -> None:
        """Inicializa um plano vazio ou o preset correspondente ao estágio I."""
        if policy not in PLAN_POLICIES:
            raise ValueError(f"policy must be one of {PLAN_POLICIES}")
        self.policy = policy
        self.tube_count: Dict[str, int] = {"short": 0, "long": 0}
        self.screw_count: Dict[str, int] = {
            "bottom": 0,
            "four_tubes": 0,
            "top": 0,
        }
        self.wheels_count = 0
        self.action_history: List[str] = []
        self.intention_history: List[str] = []
        self.stage_record: Dict[str, List[str]] = {
            "bottom": [],
            "four_tubes": [],
            "top": [],
        }
        self.stage: Optional[str] = None
        self.stage_history: List[str] = []

        if stageI_done:
            stage_i_actions = [
                "get_short_tubes",
                "get_long_tubes",
                "get_short_tubes",
                "get_long_tubes",
            ]
            self.tube_count = {"short": 2, "long": 2}
            self.screw_count["bottom"] = 4
            self.action_history = list(stage_i_actions)
            self.intention_history = ["get_connectors"] * 4
            self.stage_record["bottom"] = list(stage_i_actions)
            self.stage_history = ["bottom"]

    def apply_intention(self, intention: str) -> Optional[Tuple[str, int]]:
        """Aplica uma classe de intenção e decide a próxima ação.

        Args:
            intention: Uma chave de ``INTENTION_LIST``: ``no_action``,
                ``get_connectors``, ``get_screws`` ou ``get_wheels``.

        Returns:
            Tupla ``(nome_da_acao, indice_previo)`` quando o controlador
            enviaria uma ação, ou ``None`` quando não há ação a executar.

        Raises:
            ValueError: Se a intenção não pertence ao contrato do modelo.

        O método replica a fase de decisão. Quando o retorno não é ``None``, o
        chamador deve invocar ``apply_action(nome_da_acao)`` após a execução
        bem-sucedida para reproduzir a atualização completa do controlador.
        """
        if intention not in INTENTION_LIST:
            raise ValueError(f"unknown intention: {intention!r}")

        action = self._decide_action(intention)
        if action is not None:
            self.intention_history.append(intention)
        return action

    def apply_command(self, command: str) -> Optional[Tuple[str, int]]:
        """Aplica um comando manual reconhecido por ``decide_send_action``.

        Args:
            command: Um de ``short``, ``long``, ``spin`` ou ``get up``.

        Returns:
            A ação decidida e seu índice prévio, ou ``None``.

        Raises:
            ValueError: Se o comando não for reconhecido.

        Comandos manuais não entram em ``intention_history``, tal como no
        receptor real. Para o estágio ``four_tubes``, chame antes
        ``begin_four_tubes_stage`` e aplique quatro comandos ``short``.
        """
        if command not in _COMMANDS:
            raise ValueError(f"unknown controller command: {command!r}")
        return self._decide_action(command)

    def begin_four_tubes_stage(self) -> None:
        """Ativa o estágio manual ``four_tubes`` usado pelo receptor real.

        Raises:
            RuntimeError: Se ``bottom`` não terminou, outro estágio está ativo
                ou ``four_tubes`` já foi concluído.

        Essa ativação fica fora de ``decide_send_action`` no controlador
        original; o método a torna explícita sem incorporar ROS ou entrada de
        voz ao simulador.
        """
        if self.stage is not None:
            raise RuntimeError(f"stage already active: {self.stage}")
        if len(self.stage_record["bottom"]) != 4:
            raise RuntimeError("bottom stage must be completed first")
        if len(self.stage_record["four_tubes"]) >= 4:
            raise RuntimeError("four_tubes stage is already completed")
        self.stage = "four_tubes"

    def apply_action(self, action: str) -> None:
        """Registra uma ação concluída com sucesso.

        Args:
            action: Nome concreto retornado por ``apply_intention`` ou
                ``apply_command``.

        Raises:
            ValueError: Se o nome da ação não pertence ao PlanGraph.
            RuntimeError: Se uma ação de tubo exceder os limites físicos do
                plano.

        A atualização corresponde ao final bem-sucedido de ``execute_action``:
        incrementa tubos, registra o histórico e encerra um estágio ao atingir
        quatro ações.
        """
        if action not in _ACTIONS:
            raise ValueError(f"unknown action: {action!r}")

        tube_key: Optional[str] = None
        if action == "get_short_tubes":
            tube_key = "short"
        elif action == "get_long_tubes":
            tube_key = "long"

        if tube_key is not None:
            if self.tube_count[tube_key] >= _TUBE_LIMITS[tube_key]:
                raise RuntimeError(f"{tube_key} tube limit exceeded")
            self.tube_count[tube_key] += 1

        self.action_history.append(action)

        if self.stage is not None:
            record = self.stage_record[self.stage]
            if len(record) >= 4:
                raise RuntimeError(f"stage {self.stage!r} already has four actions")
            record.append(action)
            if len(record) == 4:
                self.stage_history.append(self.stage)
                self.stage = None

    def step(self, intention: str) -> Optional[Tuple[str, int]]:
        """Decide e confirma imediatamente uma intenção para replay offline.

        Args:
            intention: Classe pertencente a ``INTENTION_LIST``.

        Returns:
            A ação decidida, ou ``None``. Se houver ação, ela é registrada como
            concluída antes do retorno.

        Este atalho é apropriado para ``build_json.py``, onde as anotações
        representam ações observadas como concluídas. Use os métodos separados
        para simular falha ou atraso de execução.
        """
        action = self.apply_intention(intention)
        if action is not None:
            self.apply_action(action[0])
        return action

    def to_context_vector(self, dim: int = 7) -> List[float]:
        """Serializa o estado corrente em um vetor de contexto 7D ou 10D.

        Args:
            dim: ``7`` para o vetor principal ou ``10`` para a ablação.

        Returns:
            Em 7D: one-hot de ``[none, bottom, four_tubes, top]``, conectores
            coletados sobre 8, parafusos totais sobre 12 e rodas sobre 4.
            Em 10D: o mesmo one-hot, tubos curtos sobre 8, tubos longos sobre
            4, parafusos de cada estágio sobre 4 e rodas sobre 4.

        Raises:
            ValueError: Se ``dim`` não for 7 nem 10.
        """
        if dim not in CONTEXT_DIMS:
            raise ValueError(f"context dimension must be one of {CONTEXT_DIMS}")

        stage_one_hot = [
            1.0 if self.stage is candidate else 0.0 for candidate in STAGES
        ]
        wheels = self._normalize(self.wheels_count, 4)

        if dim == 7:
            connectors = self._normalize(
                self.tube_count["short"] + self.tube_count["long"],
                8,
            )
            screws = self._normalize(sum(self.screw_count.values()), 12)
            return stage_one_hot + [connectors, screws, wheels]

        return stage_one_hot + [
            self._normalize(self.tube_count["short"], 8),
            self._normalize(self.tube_count["long"], 4),
            self._normalize(self.screw_count["bottom"], 4),
            self._normalize(self.screw_count["four_tubes"], 4),
            self._normalize(self.screw_count["top"], 4),
            wheels,
        ]

    def snapshot(self) -> Dict[str, Any]:
        """Retorna uma cópia independente e serializável do estado completo."""
        return {
            "policy": self.policy,
            "tube_count": deepcopy(self.tube_count),
            "screw_count": deepcopy(self.screw_count),
            "wheels_count": self.wheels_count,
            "action_history": list(self.action_history),
            "intention_history": list(self.intention_history),
            "stage_record": deepcopy(self.stage_record),
            "stage": self.stage,
            "stage_history": list(self.stage_history),
        }

    def _decide_action(self, event: str) -> Optional[Tuple[str, int]]:
        """Replica as condições de ``decide_send_action`` do receptor."""
        if event == "get_connectors":
            if len(self.stage_record["bottom"]) < 4:
                stage = "bottom"
            elif (
                (
                    self.intention_history.count(event) >= 4
                    or "spin_bottom" in self.action_history
                )
                and self.stage != "four_tubes"
                and len(self.stage_record["top"]) < 4
            ):
                stage = "top"
            else:
                return None

            self.stage = stage
            for action, opposite_tube in _REVERT_TUBE.items():
                if self.stage_record[stage].count(action) == 2:
                    return (
                        f"get_{opposite_tube}_tubes",
                        self.tube_count[opposite_tube],
                    )

            if self.stage_record[stage]:
                last_action = self.stage_record[stage][-1]
                tube_key = _REVERT_TUBE[last_action]
            else:
                tube_key = "short"
            return (f"get_{tube_key}_tubes", self.tube_count[tube_key])

        if self.policy == "proxy_graph" and event in (
            "get_screws",
            "get_wheels",
        ):
            return self._decide_proxy_graph_action(event)

        spin_count = self._total_spin_count()
        if event == "get_wheels" and spin_count < 8:
            event = "get_screws"

        if event == "get_screws" and self.stage_history:
            if spin_count >= 10:
                event = "get_wheels"
            elif self.stage_history[-1] == "bottom":
                if self.screw_count["bottom"] < 4:
                    self.screw_count["bottom"] += 1
                    return (
                        "spin_bottom",
                        self.action_history.count("spin_bottom"),
                    )
            elif self.stage_history[-1] == "four_tubes":
                if (
                    self.screw_count["four_tubes"] < 4
                    and len(self.stage_history) == 3
                ):
                    self.screw_count["four_tubes"] += 1
                    count = self.screw_count["four_tubes"]
                    if count in (1, 3):
                        return (
                            "spin_four_tubes",
                            self.action_history.count("spin_four_tubes"),
                        )
            elif self.stage_history[-1] == "top":
                if len(self.stage_history) == 2:
                    if self.screw_count["top"] < 4:
                        self.screw_count["top"] += 1
                        return (
                            "spin_bottom",
                            self.action_history.count("spin_bottom"),
                        )
                elif len(self.stage_history) == 3:
                    if self.screw_count["top"] < 8:
                        self.screw_count["top"] += 1
                        if self.screw_count["top"] % 4 in (1, 3):
                            return (
                                "spin_top",
                                self.action_history.count("spin_top"),
                            )

        if event == "get_wheels" and self.stage_history:
            self.wheels_count += 1
            if self.wheels_count in (1, 3):
                return ("lift_up", self.action_history.count("lift_up"))

        if event == "get up":
            self.wheels_count += 1
            return ("lift_up", self.action_history.count("lift_up"))

        if event == "spin" and self.stage_history:
            stage = self.stage_history[-1]
            self.screw_count[stage] = 1
            return (
                f"spin_{stage}",
                self.action_history.count(f"spin_{stage}"),
            )

        if event in ("short", "long"):
            if self.tube_count[event] < _TUBE_LIMITS[event]:
                return (
                    f"get_{event}_tubes",
                    self.tube_count[event],
                )

        return None

    def _decide_proxy_graph_action(
        self,
        event: str,
    ) -> Optional[Tuple[str, int]]:
        """Decide ações no grafo corrigido usado pela tarefa proxy."""
        if event == "get_screws":
            for stage in reversed(self.stage_history):
                if self.screw_count[stage] < 4:
                    self.screw_count[stage] += 1
                    action = f"spin_{stage}"
                    return (action, self.action_history.count(action))
            return None

        if event == "get_wheels":
            if all(count == 4 for count in self.screw_count.values()):
                if self.wheels_count < 4:
                    self.wheels_count += 1
                    return ("lift_up", self.action_history.count("lift_up"))
            return None

        return None

    def _total_spin_count(self) -> int:
        return sum(
            self.action_history.count(action)
            for action in ("spin_bottom", "spin_four_tubes", "spin_top")
        )

    @staticmethod
    def _normalize(value: int, maximum: int) -> float:
        # O controlador legado permite alguns contadores acima do protocolo
        # experimental. O contexto permanece no contrato normalizado [0, 1].
        return min(max(float(value) / float(maximum), 0.0), 1.0)
