from __future__ import annotations

from pathlib import Path

from games.nim.adapter import NimAdapter
from rl.mcts_agent import MCTSAgent
from rl.self_play import PolicyMCTSAgent, SelfPlayTrainer


def test_self_play_trainer_completes_iterations() -> None:
    """End-to-end: trainer runs the configured number of iterations and produces
    a history entry per iteration with the documented metric keys.
    """
    adapter = NimAdapter(n_sticks=11, max_take=3)
    trainer = SelfPlayTrainer(
        adapter,
        n_iterations=2,
        n_games_per_iter=8,
        eval_games=8,
        mcts_simulations=20,
        verbose=False,
        seed=0,
    )
    agent, history = trainer.train()

    assert isinstance(agent, MCTSAgent)
    assert len(history) == 2
    required = {"iteration", "candidate_win_rate", "promoted", "policy_unique_states"}
    for entry in history:
        assert required <= entry.keys()
        assert 0.0 <= entry["candidate_win_rate"] <= 1.0
        assert entry["policy_unique_states"] > 0  # self-play visited at least one state


def test_save_load_roundtrip_preserves_a_working_agent(tmp_path: Path) -> None:
    adapter = NimAdapter(n_sticks=7, max_take=3)
    trainer = SelfPlayTrainer(
        adapter,
        n_iterations=1,
        n_games_per_iter=5,
        eval_games=4,
        mcts_simulations=10,
        verbose=False,
        seed=1,
    )
    trainer.train()

    path = tmp_path / "agent.pkl"
    trainer.save_agent(path)
    assert path.exists() and path.stat().st_size > 0

    fresh = SelfPlayTrainer(
        adapter,
        n_iterations=1,
        n_games_per_iter=5,
        eval_games=4,
        mcts_simulations=10,
        verbose=False,
    )
    loaded = fresh.load_agent(path)
    assert isinstance(loaded, MCTSAgent)

    # Loaded agent must be usable end-to-end.
    state = adapter.get_initial_state()
    obs = adapter.get_observable_state(state, 0)
    legal = adapter.get_legal_actions(state, 0)
    loaded.on_game_start(0, adapter.get_n_players())
    chosen = loaded.choose_action(obs, legal)
    assert chosen in legal


def test_empty_policy_mcts_behaves_like_vanilla_mcts() -> None:
    """A PolicyMCTSAgent with no policy data must remain a legal MCTS agent
    (rollouts degenerate to uniform Laplace weights of 0.5).
    """
    adapter = NimAdapter(n_sticks=9)
    agent = PolicyMCTSAgent(adapter, policy={}, n_simulations=20, seed=0)
    agent.on_game_start(0, 2)
    state = adapter.get_initial_state()
    obs = adapter.get_observable_state(state, 0)
    legal = adapter.get_legal_actions(state, 0)
    chosen = agent.choose_action(obs, legal)
    assert chosen in legal
