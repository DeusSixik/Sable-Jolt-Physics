# Sable Jolt Physics

A mod that replaces the physics engine in [Sable](https://github.com/ryanhcode/sable)
from [Rapier](https://github.com/dimforge/rapier) to [Jolt Physics](https://github.com/jrouwe/joltphysics).

## Why

In complex scenes, Rapier caused severe performance drops, which made the server TPS fall. After switching to
Jolt, the load dropped noticeably in our scenarios.

## Performance

### Test machine

- CPU: Intel Core i7-14700KF
- RAM: 64 GB (3200 MHz), 16 GB allocated to Minecraft
- GPU: RTX 5070
- Java: GraalVM JDK 25.0.4+7.1

Tick time was measured with the `/tick query` command.

### 1,540 falling objects with collisions (Min, Max)

| Phase                         | Rapier (MSPT) | Jolt (MSPT) |
|-------------------------------|---------------|-------------|
| During the fall               | 180.0-248.5   | 67.3-210.2  |
| After landing (idle)          | 48.0-54.4     | 25.6-35.5   |

### 400 hovering panels rotating 6° every 12 ticks (Min, Max)

| Rapier (MSPT) | Jolt (MSPT) |
|---------------|-------------|
| 10.7-18.9     | 6.1-13.4    |

## Known limitations

> [!WARNING]
> The project is in Alpha; stable operation is not guaranteed. Please report any issues
> in [Issues](https://github.com/DeusSixik/Sable-Jolt-Physics/issues).

> [!CAUTION]
> Sable's behavior with Jolt may differ from its behavior with Rapier.

## Acknowledgements

- [DenisMasterHerobrine](https://github.com/DenisMasterHerobrine) - fixes and ARM support.

## License

- [GPLv3](LICENSE)