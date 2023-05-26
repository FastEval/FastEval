import { createBenchmarksV } from './benchmarks/main.js'

async function main() {
    window.addEventListener('hashchange', () => {
        location.reload()
    })

    const url = location.hostname === 'tju01.github.io'
        ? 'https://raw.githubusercontent.com/tju01/oasst-automatic-model-eval/main/reports'
        : './reports'

    const benchmarksV = await createBenchmarksV(url)
    document.body.appendChild(benchmarksV)
}

main()
