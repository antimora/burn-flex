use std::time::Instant;
use tracel_xtask::prelude::*;

#[macros::base_commands(
    Bump,
    Check,
    Compile,
    Coverage,
    Doc,
    Dependencies,
    Fix,
    Publish,
    Validate,
    Vulnerabilities
)]
pub enum Command {
    /// Build burn-flex.
    Build(BuildCmdArgs),
    /// Test burn-flex.
    Test(TestCmdArgs),
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();
    let (args, environment) = init_xtask::<Command>(parse_args::<Command>()?)?;

    match args.command {
        Command::Build(cmd_args) => {
            base_commands::build::handle_command(cmd_args, environment, args.context)
        }
        Command::Test(cmd_args) => {
            base_commands::test::handle_command(cmd_args, environment, args.context)
        }
        _ => dispatch_base_commands(args, environment),
    }?;

    let duration = start.elapsed();
    println!(
        "\x1B[32;1mTime elapsed: {}\x1B[0m",
        format_duration(&duration)
    );

    Ok(())
}
