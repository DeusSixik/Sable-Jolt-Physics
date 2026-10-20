<p align="center">
  <img src="./.idea/icon.png" alt="Sable Jolt Physics logo" width="160">
</p>

<h1 align="center">Sable Jolt Physics</h1>

<p align="center">
  Replaces the physics engine in <a href="https://github.com/ryanhcode/sable">Sable</a>
  with <a href="https://github.com/jrouwe/joltphysics">Jolt Physics</a> instead of
  <a href="https://github.com/dimforge/rapier">Rapier</a>.
</p>

<!--
<p align="center">
  <a href="https://modrinth.com/mod/sable-jolt-physics">
    <img src="https://img.shields.io/modrinth/dt/sable-jolt-physics?logo=modrinth&label=&suffix=%20&style=flat&color=242629&labelColor=5CA424&logoColor=1C1C1C" alt="Modrinth">
  </a>
  <a href="https://www.curseforge.com/minecraft/mc-mods/sable-jolt-physics">
    <img src="https://img.shields.io/curseforge/dt/<YOUR_PROJECT_ID>?logo=curseforge&label=&suffix=%20&style=flat&color=242629&labelColor=F16436&logoColor=1C1C1C" alt="CurseForge">
  </a>
</p>
-->

---

## Why

In complex scenes, Rapier caused severe performance drops, which made the server TPS fall.
After switching to Jolt, the load dropped noticeably in our scenarios.

## Performance

<details>
<summary><b>Test machine</b></summary>

| Component | Value                                          |
|-----------|------------------------------------------------|
| CPU       | Intel Core i7-14700KF                          |
| RAM       | 64 GB (3200 MHz), 16 GB allocated to Minecraft |
| GPU       | RTX 5070                                       |
| Java      | GraalVM JDK 25.0.4+7.1                         |

Tick time was measured with `/tick query`. Values are min-max.

</details>

### 1,540 falling objects with collisions

| Phase                | Rapier (MSPT) | Jolt (MSPT) |
|----------------------|:-------------:|:-----------:|
| During the fall      |  180.0-248.5  | 67.3-210.2  |
| After landing (idle) |   48.0-54.4   |  25.6-35.5  |

### 400 hovering panels rotating 6° every 12 ticks

| Rapier (MSPT) | Jolt (MSPT) |
|:-------------:|:-----------:|
|   10.7-18.9   |  6.1-13.4   |

## Known limitations

> [!WARNING]
> The project is in Alpha; stable operation is not guaranteed.
> Please report any issues in [Issues](https://github.com/DeusSixik/Sable-Jolt-Physics/issues).

> [!CAUTION]
> Sable's behavior with Jolt may differ from its behavior with Rapier.

## Acknowledgements

- [DenisMasterHerobrine](https://github.com/DenisMasterHerobrine) — fixes and ARM support.

## License

Distributed under [GPLv3](LICENSE).