import { createBenchmarksV } from './benchmarks/main.js'

async function main() {
    window.addEventListener('hashchange', () => {
        location.reload()
    })

    const benchmarksV = await createBenchmarksV('https://raw.githubusercontent.com/tju01/oasst-automatic-model-eval/main/reports')
    document.body.appendChild(benchmarksV)
}

main()
