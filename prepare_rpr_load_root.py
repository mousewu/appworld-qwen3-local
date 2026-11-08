"""Create the paired actor/critic Megatron load roots expected by RPR."""
import argparse
from pathlib import Path


def prepare(actor_iter, critic_iter, output):
    actor_iter = Path(actor_iter).expanduser().resolve(strict=True)
    critic_iter = Path(critic_iter).expanduser().resolve(strict=True) if critic_iter else None
    output = Path(output).expanduser().resolve()
    critic_output = Path(str(output) + '_critic')
    if not actor_iter.is_dir() or (critic_iter is not None and not critic_iter.is_dir()):
        raise ValueError('Actor and critic iteration paths must be directories')
    if output.exists() or critic_output.exists():
        raise FileExistsError(f'Refusing to replace {output} or {critic_output}')
    output.mkdir(parents=True)
    (output/'release').symlink_to(actor_iter, target_is_directory=True)
    (output/'latest_checkpointed_iteration.txt').write_text('release\n')
    if critic_iter is not None:
        critic_output.mkdir(parents=True)
        (critic_output/'release').symlink_to(critic_iter, target_is_directory=True)
        (critic_output/'latest_checkpointed_iteration.txt').write_text('release\n')
    return output, critic_output if critic_iter is not None else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor-iter', required=True, type=Path)
    parser.add_argument('--critic-iter', type=Path,
                        help='Omit for a cold Critic warm-up; required for RPR-1/RPR-2')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    actor, critic = prepare(args.actor_iter, args.critic_iter, args.output)
    print('actor :', actor/'release', '->', (actor/'release').resolve())
    print('critic:', (str(critic/'release')+' -> '+str((critic/'release').resolve()))
          if critic else 'none (cold value head)')


if __name__ == '__main__':
    main()
